import torch
import torch.nn as nn
from resnet import resnext50_32x4d


class Rectifier(nn.Module):
    def __init__(self):
        super(Rectifier, self).__init__()

        self.encoder = resnext50_32x4d()
        self.xencoder = resnext50_32x4d()
        del self.encoder.avgpool
        del self.encoder.fc

        self.decoder = Decoder(False)
        self.xdecoder = Decoder(True)

    def forward(self, x, xs):
        xc1 = self.xencoder.relu(self.xencoder.bn1(self.xencoder.conv1(x)))
        c1 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(xs)))
        c2 = self.encoder.layer1(self.encoder.maxpool(c1))
        c3 = self.encoder.layer2(c2)
        c4 = self.encoder.layer3(c3)
        c5 = self.encoder.layer4(c4)

        f1 = self.decoder.layer1(c5)
        f2 = self.decoder.layer2(torch.cat((f1, c4), 1))
        f3 = self.decoder.layer3(torch.cat((f2, c3), 1))
        f4 = self.decoder.resblock1(f3)
        f5 = self.decoder.resblock2(f4)
        f6 = self.decoder.layer4(torch.cat((f5, c2), 1))
        f7 = self.decoder.layer5(torch.cat((f6, c1), 1))

        xc2 = self.xencoder.layer1(self.xencoder.maxpool(f7+xc1))
        xc3 = self.xencoder.layer2(xc2)
        xc4 = self.xencoder.layer3(xc3)
        xc5 = self.xencoder.layer4(xc4)

        xf1 = self.xdecoder.layer1(xc5)
        xf2 = self.xdecoder.layer2(torch.cat((xf1, xc4), 1))
        xf3 = self.xdecoder.layer3(torch.cat((xf2, xc3), 1))
        xf4 = self.xdecoder.resblock1(xf3)
        xf5 = self.xdecoder.resblock2(xf4)
        xf6 = self.xdecoder.layer4(torch.cat((xf5, xc2), 1))
        xf7 = self.xdecoder.layer5(torch.cat((xf6, xc1), 1))

        out = self.decoder.tanh(xf7)

        return out


class Decoder(nn.Module):
    def __init__(self, xtype):
        super(Decoder, self).__init__()
        self.xtype = xtype
        self.layer1 = self._make_layer(1)
        self.layer2 = self._make_layer(2)
        self.layer3 = self._make_layer(3)
        self.resblock1 = ResidualBlock(256)
        self.resblock2 = ResidualBlock(256)
        self.layer4 = self._make_layer(4)
        self.layer5 = self._make_layer(5)
        self.tanh = nn.Tanh()


    def _make_layer(self, n):
        in_ngf = [2048, 1024 * 2, 512 * 2, 256 * 2, 64 * 2]
        out_ngf = [1024, 512, 256, 64, 1]
        if n == 1 or n == 2 or (not self.xtype and n == 3):
            dconv = nn.ConvTranspose2d(in_ngf[n-1], out_ngf[n-1], kernel_size=(4, 3), stride=2, padding=1)
        else:
            dconv = nn.ConvTranspose2d(in_ngf[n-1], out_ngf[n-1], kernel_size=4, stride=2, padding=1)
        bn = nn.BatchNorm2d(out_ngf[n-1])
        elu = nn.ELU()
        layers = []
        layers.append(dconv)
        if n != 5:
            layers.append(bn)
            layers.append(elu)
        if n == 1 or n == 2:
            dropout = nn.Dropout()
            layers.append(dropout)

        return nn.Sequential(*layers)

    def forward(self, input):
        f1 = self.layer1(input)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.decoder.resblock1(f3)
        f5 = self.decoder.resblock2(f4)
        f6 = self.layer4(f5)
        f7 = self.layer5(f6)
        out = self.tanh(f7)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.elu = nn.ELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.elu(residual)
        residual = self.conv2(residual)
        residual = x + residual
        out = self.elu(residual)

        return out
