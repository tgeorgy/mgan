import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim

from models import Generator, Discriminator
from utils import CelebADatasetLoader


cudnn.benchmark = True
batch_size = 64
n_latent = 16
torch.manual_seed(1)
np.random.seed(1)


netG = Generator(n_latent).cuda()
netD = Discriminator(n_attr).cuda()

criterion_MSE = nn.MSELoss().cuda()
criterion_L1 = nn.L1Loss().cuda()
#criterion_AE = nn.MSELoss().cuda()

# Init tensors
inputG_gpu = torch.FloatTensor(batch_size, 3, 128, 128).cuda()
inputD_gpu = torch.FloatTensor(batch_size, 3, 128, 128).cuda()
latent_gpu = torch.FloatTensor(batch_size, n_latent).cuda()

# Convert to Variables
inputG_gpu = Variable(inputG_gpu)
inputD_gpu = Variable(inputD_gpu)
latent_gpu = Variable(latent_gpu)

dl = CelebADatasetLoader(batch_size, n_latent)

optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas = (0.5, 0.999))

print_every = 200
n_epochs = 60
margin = 0.015
AE_coef = 0.15

netD.train()
netG.train()

for epoch_i in xrange(n_epochs):
    batch_i = 0
    for inputG, inputD, latent in dl:
        inputG_gpu.data.copy_(inputG)
        inputD_gpu.data.copy_(inputD)
        latent_gpu.data.copy_(latent)

        # Train Discriminator
        netD.zero_grad()
        netG.zero_grad()

        output = netD(inputD_gpu)
        loss_D_real = criterion_MSE(output, inputD_gpu)
        loss_D_real.backward(retain_variables=True)

        fake, mask = netG(inputG_gpu, latent_gpu)
        output = netD(fake.detach())
        loss_D_fake = margin - criterion_MSE(output, fake.detach())
        loss_D_fake = loss_D_fake.clamp(min=0)
        loss_D_fake.backward()

        loss_D = (loss_D_real + loss_D_fake) / 2
        optimizerD.step()


        # Train Generator
        output, features = netD(fake)
        loss_G = (output - fake).pow(2).mean()  # MSE

        loss_AE = criterion_L1(fake,inputG_gpu)

        loss_G = loss_G + AE_coef*loss_AE
        loss_G.backward()

        optimizerG.step()

        batch_i += 1
        if batch_i % print_every == 0 and batch_i > 1:
            print('Epoch #%d' % (epoch_i+1))
            print('Batch #%d' % batch_i)
            print('loss_D: %0.3f'%loss_D.data[0] + '\tloss_G: %0.3f'%loss_G.data[0])
            print('Loss D real: %0.3f'%loss_D_real.data[0], 'Loss D fake: %0.3f'%loss_D_fake.data[0])

            print('Loss AE: %0.3f'%loss_AE.data[0])

            torchvision.utils.save_image(torch.cat([fake.data.cpu()[:8], inputG[:8]]), 'progress.png', nrow=8, padding=1)
