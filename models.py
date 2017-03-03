import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class GeneratorEnc(nn.Module):
    def __init__(self):
        super(GeneratorEnc, self).__init__()
        # Encoder layers
        self.enc_conv1 = nn.Conv2d(3, 16, 3, 2, 1)
        self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.enc_bn4 = nn.BatchNorm2d(128)
        self.enc_conv5 = nn.Conv2d(128, 128, 3, 2, 1)
        self.enc_bn5 = nn.BatchNorm2d(128)

    def forward(self, img):
        enc = F.leaky_relu(self.enc_bn1(self.enc_conv1(img)), 0.2)
        enc = F.leaky_relu(self.enc_bn2(self.enc_conv2(enc)), 0.2)
        enc = F.leaky_relu(self.enc_bn3(self.enc_conv3(enc)), 0.2)
        enc = F.leaky_relu(self.enc_bn4(self.enc_conv4(enc)), 0.2)
        enc = F.leaky_relu(self.enc_bn5(self.enc_conv5(enc)), 0.2)

        return enc

class GeneratorDec(nn.Module):
    def __init__(self):
        super(GeneratorDec, self).__init__()

        # Latent parameters embedding layers
        #self.embed_fc1 = nn.Linear(nlatent, 128)
        #self.embed_bn1 = nn.BatchNorm1d(128)
        #self.embed_fc2 = nn.Linear(128, 64*4*4)
        #self.embed_bn2 = nn.BatchNorm2d(64)

        # Decoder layers
        self.dec_conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.dec_bn1 = nn.BatchNorm2d(128)
        self.dec_conv1_ = nn.Conv2d(128, 128*4, 1, 1, 0, bias=False)
        self.dec_bn1_ = nn.BatchNorm2d(128)

        self.dec_conv2 = nn.Conv2d(128+3, 128, 3, 1, 1, bias=False)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_conv2_ = nn.Conv2d(128, 128*4, 1, 1, 0, bias=False)
        self.dec_bn2_ = nn.BatchNorm2d(128)

        self.dec_conv3 = nn.Conv2d(128+3, 64, 3, 1, 1, bias=False)
        self.dec_bn3 = nn.BatchNorm2d(64)
        self.dec_conv3_ = nn.Conv2d(64, 64*4, 1, 1, 0, bias=False)
        self.dec_bn3_ = nn.BatchNorm2d(64)

        self.dec_conv4 = nn.Conv2d(64+3, 64, 3, 1, 1, bias=False)
        self.dec_bn4 = nn.BatchNorm2d(64)
        self.dec_conv4_ = nn.Conv2d(64, 64*4, 1, 1, 0, bias=False)
        self.dec_bn4_ = nn.BatchNorm2d(64)

        self.dec_conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.dec_conv5_ = nn.Conv2d(32, 3*4, 1, 1, 0, bias=False)

        # Gate layers
        self.gate_conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.gate_conv5_ = nn.Conv2d(32, 4, 1, 1, 0, bias=False)
        self.gate_bn5 = nn.BatchNorm2d(32)

    def forward(self, enc, img):
        #net_emb = F.leaky_relu(self.embed_bn1(self.embed_fc1(embed)), 0.2)
        #net_emb = self.embed_fc2(net_emb)
        #net_emb = net_emb.view(-1, 64, 4, 4)
        #net_emb = self.embed_bn2(net_emb)

        #x = torch.cat([enc, net_emb], 1)
        dec = F.leaky_relu(self.dec_bn1(self.dec_conv1(enc)), 0.2)
        dec = F.pixel_shuffle(self.dec_conv1_(dec), 2)
        dec = F.leaky_relu(self.dec_bn1_(dec), 0.2)

        dec = torch.cat([dec, F.avg_pool2d(img, 16)], 1)  # skip connection
        dec = F.leaky_relu(self.dec_bn2(self.dec_conv2(dec)), 0.2)
        dec = F.pixel_shuffle(self.dec_conv2_(dec), 2)
        dec = F.leaky_relu(self.dec_bn2_(dec), 0.2)

        dec = torch.cat([dec, F.avg_pool2d(img, 8)], 1)  # skip connection
        dec = F.leaky_relu(self.dec_bn3(self.dec_conv3(dec)), 0.2)
        dec = F.pixel_shuffle(self.dec_conv3_(dec), 2)
        dec = F.leaky_relu(self.dec_bn3_(dec), 0.2)

        dec = torch.cat([dec, F.avg_pool2d(img, 4)], 1)  # skip connection
        dec = F.leaky_relu(self.dec_bn4(self.dec_conv4(dec)), 0.2)
        dec = F.pixel_shuffle(self.dec_conv4_(dec), 2)
        dec = F.leaky_relu(self.dec_bn4_(dec), 0.2)

        gate = F.leaky_relu(self.gate_bn5(self.gate_conv5(dec)), 0.2)
        gate = F.pixel_shuffle(self.gate_conv5_(gate), 2)
        gate = gate.repeat(1, 3, 1, 1)

        dec = F.leaky_relu(self.dec_conv5(dec), 0.2)
        dec = F.pixel_shuffle(self.dec_conv5_(dec), 2)

        gate = F.sigmoid(gate)*0.98 + 0.01  # Adding leak

        dec = gate*dec + (1-gate)*img
        dec = torch.clamp(dec, -1, 1)  # just in case

        return dec, gate


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Encoder layers
        self.enc_conv1 = nn.Conv2d(3, 16, 4, 2, 1, bias=False)
        self.enc_bn1 = nn.BatchNorm2d(16)  # 64x64
        self.enc_conv2 = nn.Conv2d(16, 32, 4, 2, 1, bias=False)
        self.enc_bn2 = nn.BatchNorm2d(32)  # 32x64
        self.enc_conv3 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.enc_bn3 = nn.BatchNorm2d(64)  # 16x16
        self.enc_conv4 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.enc_bn4 = nn.BatchNorm2d(128)  # 8x8
        self.enc_conv5 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.enc_bn5 = nn.BatchNorm2d(256)  # 4x4
        self.enc_conv6 = nn.Conv2d(256, 256, 4, 2, 1, bias=False)
        self.enc_bn6 = nn.BatchNorm2d(256)  # 2x2

        self.fc = nn.Linear(1024, 1)


    def forward(self, img):
        x = F.leaky_relu(self.enc_bn1(self.enc_conv1(img)), 0.2)
        x = F.leaky_relu(self.enc_bn2(self.enc_conv2(x)), 0.2)
        x = F.leaky_relu(self.enc_bn3(self.enc_conv3(x)), 0.2)
        x = F.leaky_relu(self.enc_bn4(self.enc_conv4(x)), 0.2)
        x = F.leaky_relu(self.enc_bn5(self.enc_conv5(x)), 0.2)
        x = F.leaky_relu(self.enc_bn6(self.enc_conv6(x)), 0.2)

        x = x.view(-1, 256*2*2)
        x = self.fc(x)

        return x
