from utils import CelebADatasetLoader  # has to load cv2 before loading torch

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torchvision.utils import save_image

from models import GeneratorEnc, GeneratorDec, Discriminator


parser = argparse.ArgumentParser()
parser.add_argument('--coefGAN', default=1.0, type=float, help='GAN loss weight')
parser.add_argument('--coefTID', default=1.0, type=float, help='TID loss weight')
parser.add_argument('--coefCONST', default=1.0, type=float, help='CONST loss weight')

parser.add_argument('--logdir', default='000', type=str)

opt = parser.parse_args()
print(opt)

odir = os.path.join('ckpt', opt.logdir)
if not os.path.exists(odir):
    os.mkdir(odir)

cudnn.benchmark = True
batch_size = 64
n_latent = 16

torch.manual_seed(1)
np.random.seed(1)

print('Initializing models')
netGE = GeneratorEnc().cuda()
netGD = GeneratorDec().cuda()
netD = Discriminator().cuda()

criterion_MSE = nn.MSELoss().cuda()
criterion_L1 = nn.L1Loss().cuda()

# Init tensors
inputG_gpu = torch.FloatTensor(batch_size, 3, 128, 128).cuda()
inputD_gpu = torch.FloatTensor(batch_size, 3, 128, 128).cuda()
A = 1  * torch.ones(batch_size).cuda()
B = -1 * torch.ones(batch_size).cuda()
C = 1  * torch.ones(batch_size).cuda()

# Convert to Variables
inputG_gpu = Variable(inputG_gpu)
inputD_gpu = Variable(inputD_gpu)
A = Variable(A)
B = Variable(B)
C = Variable(C)

print('Loading data')
dl = CelebADatasetLoader(batch_size, n_latent)

optimizerGE = optim.Adam(netGE.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizerGD = optim.Adam(netGD.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))

print_every = 200
n_epochs = 60

netD.train()
netGE.train()
netGD.train()

print('Training...')
for epoch_i in xrange(n_epochs):
    batch_i = 0
    for inputG, inputD, _ in dl:
        inputG_gpu.data.copy_(inputG)
        inputD_gpu.data.copy_(inputD)

        # Train Discriminator
        netD.zero_grad()
        netGE.zero_grad()
        netGD.zero_grad()

        output = netD(inputD_gpu)
        loss_D_real = criterion_MSE(output, A)
        loss_D_real.backward(retain_variables=True)

        features_G = netGE(inputG_gpu)
        fake, mask = netGD(features_G, inputG_gpu)
        output = netD(fake.detach())
        loss_D_fake = criterion_MSE(output, B)
        loss_D_fake.backward()

        loss_D = (loss_D_real + loss_D_fake) / 2
        optimizerD.step()

        # Train Generator
        output = netD(fake)
        loss_GAN = criterion_MSE(output, C)

        features_target = netGE(inputD_gpu)
        reconstruction, mask = netGD(features_target, inputD_gpu)
        loss_TID = criterion_MSE(reconstruction, inputD_gpu)

        features_G2 = netGE(fake)
        loss_CONST = (features_G2 - features_G).pow(2).mean()

        loss_G = opt.coefGAN*loss_GAN
        loss_G += opt.coefTID*loss_TID
        loss_G += opt.coefCONST*loss_CONST
        loss_G.backward()

        optimizerGE.step()
        optimizerGD.step()

        batch_i += 1
        if batch_i % print_every == 0 and batch_i > 1:
            print('Epoch #%d' % (epoch_i+1))
            print('Batch #%d' % batch_i)
            print('Loss D: %0.3f' % loss_D.data[0] + '\t' +
                  'Loss G: %0.3f' % loss_G.data[0])
            print('Loss D real: %0.3f' % loss_D_real.data[0] + '\t' +
                  'Loss D fake: %0.3f' % loss_D_fake.data[0])

            print('Loss TID: %0.3f' % loss_TID.data[0])
            print('-'*50)

    epoch_i_str = str(epoch_i).zfill(3)

    save_image(torch.cat([fake.data.cpu()[:16], inputG[:16]]),
               os.path.join(odir, 'progress_'+epoch_i_str+'.png'), nrow=16, padding=1)

    torch.save(netGE, os.path.join(odir, 'netGE_'+epoch_i_str+'.pth'))
    torch.save(netGD, os.path.join(odir, 'netGD_'+epoch_i_str+'.pth'))
    torch.save(netD, os.path.join(odir, 'netD_'+epoch_i_str+'.pth'))
