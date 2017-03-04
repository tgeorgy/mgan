from utils import CelebADatasetLoader

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torchvision.utils import save_image

from models import GeneratorEnc, GeneratorDec, Discriminator


cudnn.benchmark = True
batch_size = 64
n_latent = 16
odir = 'ckpt'
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
margin = 0.02
TID_coef = 1.0

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

        features = netGE(inputG_gpu)
        fake, mask = netGD(features, inputG_gpu)
        output = netD(fake.detach())
        loss_D_fake = criterion_MSE(output, B)
        loss_D_fake.backward()

        loss_D = (loss_D_real + loss_D_fake) / 2
        optimizerD.step()

        # Train Generator
        output = netD(fake)
        loss_G = criterion_MSE(output, C)

        features_target = netGE(inputD_gpu)
        reconstruction, mask = netGD(features_target, inputD_gpu)

        loss_TID = criterion_MSE(reconstruction, inputD_gpu)

        loss_G = loss_G + TID_coef*loss_TID
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

            save_image(torch.cat([fake.data.cpu()[:8], inputG[:8]]),
                       'progress.png', nrow=8, padding=1)

            torch.save(netGE, odir+'/netGE.pth')
            torch.save(netGD, odir+'/netGD.pth')
            torch.save(netD, odir+'/netD.pth')
