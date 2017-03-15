from utils import CelebADatasetLoader  # have to load cv2 before loading torch

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

# Init parameters
parser = argparse.ArgumentParser()
parser.add_argument('--coefGAN', default=1, type=float, help='GAN loss weight')
parser.add_argument('--coefTID', default=1.0, type=float, help='TID loss weight')
parser.add_argument('--coefCONST', default=100.0, type=float, help='CONST loss weight')
parser.add_argument('--coefL1', default=0.3, type=float, help='Generator Input-Output L1 dist')
parser.add_argument('--nepochs', default=120, type=int, help='Number of epochs')
parser.add_argument('--dropout', default=0.5, type=float, help='D dropout prob')

parser.add_argument('--logdir', default='000', type=str)

opt = parser.parse_args()
print(opt)

odir = os.path.join('ckpt', opt.logdir)
if not os.path.exists(odir):
    os.mkdir(odir)

cudnn.benchmark = True
print_every = 200
save_sample_every = 500
batch_size = 64

torch.manual_seed(1)
np.random.seed(1)

print('Initializing models')
netGE = GeneratorEnc().cuda()
netGD = GeneratorDec().cuda()
netD = Discriminator(opt.dropout).cuda()

optimizerGE = optim.Adam(netGE.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
optimizerGD = optim.Adam(netGD.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)

criterion_MSE = nn.MSELoss().cuda()
criterion_L1 = nn.L1Loss().cuda()

print('Initializing Variables')
inputG_gpu = torch.FloatTensor(batch_size, 3, 128, 128).cuda()
inputG_gpu_fixed = torch.FloatTensor(batch_size, 3, 128, 128).cuda()
inputD_gpu = torch.FloatTensor(batch_size, 3, 128, 128).cuda()
P_true = torch.zeros(batch_size, 3).cuda()
P_true[:, 0] = 1
P_false = torch.zeros(batch_size, 3).cuda()
P_false[:, 1] = 1
P_fake = torch.zeros(batch_size, 3).cuda()
P_fake[:, 2] = 1

inputG_gpu = Variable(inputG_gpu)
inputG_gpu_fixed = Variable(inputG_gpu_fixed)
inputD_gpu = Variable(inputD_gpu)
P_true = Variable(P_true)
P_false = Variable(P_false)
P_fake = Variable(P_fake)

print('Loading data')
dl = CelebADatasetLoader(batch_size)

inputG, _ = dl.__iter__().next()
inputG_gpu_fixed.data.copy_(inputG)


print('Training...')
for epoch_i in xrange(opt.nepochs):
    batch_i = 0
    for inputG, inputD in dl:
        inputG_gpu.data.copy_(inputG)
        inputD_gpu.data.copy_(inputD)

        # Train Discriminator
        netD.train()
        netGE.train()
        netGD.train()

        netD.zero_grad()
        netGE.zero_grad()
        netGD.zero_grad()

        output = netD(inputD_gpu)
        loss_D_real = criterion_MSE(output, P_true)
        loss_D_real.backward(retain_variables=True)

        output = netD(inputG_gpu)
        loss_D_false = criterion_MSE(output, P_false)
        loss_D_false.backward(retain_variables=True)

        features_G = netGE(inputG_gpu)
        fake, mask = netGD(features_G, inputG_gpu)
        output = netD(fake.detach())
        loss_D_fake = criterion_MSE(output, P_false)
        loss_D_fake.backward()

        loss_D = (loss_D_real + loss_D_false + loss_D_fake) / 3
        optimizerD.step()

        # Train Generator
        output = netD(fake)
        loss_GAN = criterion_MSE(output, P_true)

        features_target = netGE(inputD_gpu)
        reconstruction, mask = netGD(features_target, inputD_gpu)
        loss_TID = criterion_MSE(reconstruction, inputD_gpu)

        features_G2 = netGE(fake)
        loss_CONST = (features_G2 - features_G).pow(2).mean()

        loss_L1 = criterion_L1(fake, inputG_gpu)

        loss_G = opt.coefGAN*loss_GAN
        loss_G += opt.coefTID*loss_TID
        loss_G += opt.coefCONST*loss_CONST
        loss_G += opt.coefL1*loss_L1
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

        if batch_i % save_sample_every == 0 and batch_i > 1:
            netGE.eval()
            netGD.eval()

            features_G = netGE(inputG_gpu_fixed)
            fake, _ = netGD(features_G, inputG_gpu_fixed)

            oimg_path = 'evolution_' + str(epoch_i).zfill(3)
            oimg_path += '_'+str(batch_i).zfill(4)
            oimg_path += '.png'
            oimg_path = os.path.join(odir, oimg_path)

            save_image(torch.cat([fake.data.cpu()[:], inputG_gpu_fixed.data.cpu()[:]]),
                       oimg_path, nrow=64, padding=1)

    epoch_i_str = str(epoch_i).zfill(3)

    save_image(torch.cat([fake.data.cpu()[:16], inputG[:16]]),
               os.path.join(odir, 'progress_'+epoch_i_str+'.png'), nrow=16, padding=1)

    torch.save(netGE, os.path.join(odir, 'netGE_'+epoch_i_str+'.pth'))
    torch.save(netGD, os.path.join(odir, 'netGD_'+epoch_i_str+'.pth'))
    torch.save(netD, os.path.join(odir, 'netD_'+epoch_i_str+'.pth'))
