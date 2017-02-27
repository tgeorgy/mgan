import cv2
import numpy as np
import h5py
import torch

class CelebADatasetLoader():
    def __init__(self, batch_size, n_latent):
        with h5py.File('cache/train.h5', 'r') as hf:
            self.img = hf['img'][:]
            self.attrs = hf['attrs'][:,0]

        tmp_range = np.arange(self.attrs.shape[0])
        self.ones = tmp_range[self.attrs == 1]
        self.zeros = tmp_range[self.attrs == 0]
        self.batch_size = batch_size
        self.n_latent = n_latent

    def augment(self, input):
        input = np.transpose(input, [0, 2, 3, 1])
        rows, cols = input.shape[1:-1]
        for i, img in enumerate(input):
            rand_ang = np.random.rand()*10-5
            Mrot = cv2.getRotationMatrix2D((cols/2,rows/2),rand_ang,1)

            rand_scales = 1 + np.random.rand(2)*0.1-0.05
            Mscale = np.float32([
                [rand_scales[0],  0,               cols*(1-rand_scales[0])/2],
                [0,               rand_scales[1],  rows*(1-rand_scales[1])/2],
                [0,               0,               1                        ],
            ])

            M = Mrot.dot(Mscale)

            input[i] = cv2.warpAffine(img, M, (cols, rows))
        
        return input


    def __iter__(self):
        self.samples_remaining = min(self.ones.shape[0], self.zeros.shape[0])
        self.shuffle_ones = np.random.permutation(self.ones)
        self.shuffle_zeros = np.random.permutation(self.zeros)
        self.start_id = 0
        return self


    def next(self):
        if self.samples_remaining < self.batch_size:
            raise StopIteration
        # Init
        latent = np.random.randn(self.batch_size, self.n_latent)
        latent = torch.from_numpy(latent)

        # Sampling
        batch_ids_ones = self.shuffle_ones[self.start_id:self.start_id + self.batch_size]
        batch_ids_zeros = self.shuffle_zeros[self.start_id:self.start_id + self.batch_size]
        input_G = self.img[batch_ids_zeros]
        input_D = self.img[batch_ids_ones]

        # Augmentation
        input_G = self.augment(input_G)
        input_D = self.augment(input_D)

        input_G = np.transpose(input_G, [0, 3, 1, 2])
        input_D = np.transpose(input_D, [0, 3, 1, 2])
        input_G = np.float32(input_G)/255*2-1
        input_D = np.float32(input_D)/255*2-1
        input_G = torch.from_numpy(input_G)
        input_D = torch.from_numpy(input_D)

        self.start_id = self.start_id + self.batch_size
        self.samples_remaining -= self.batch_size

        return input_G, input_D, latent
