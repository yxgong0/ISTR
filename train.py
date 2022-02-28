from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
import time

from models.Discriminator import Discriminator
from models.Rectifier import Rectifier
from dataset import dataset
from dataset import val_dataset
from sample_loss import SampleLoss
from models.Losses import tv_loss

parser = argparse.ArgumentParser(description='train model')
parser.add_argument('--batchSize', type=int, default=64, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
parser.add_argument('--loadSize_w', type=int, default=200, help='scale image to this size')
parser.add_argument('--loadSize_h', type=int, default=64, help='scale image to this size')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--netG', type=str, default='', help='path to pre-trained netG')
parser.add_argument('--netD', type=str, default='', help='path to pre-trained netD')
parser.add_argument('--dataPath', default='', help='path to training images')
parser.add_argument('--outf', default='./results/', help='folder to output images and model checkpoints')
parser.add_argument('--val_path', default='', help='path to val images')
parser.add_argument('--val_epoch', default=50, help='path to val images')
parser.add_argument('--save_epoch', default=10, help='path to val images')
parser.add_argument('--test_step', default=4096, help='path to val images')
parser.add_argument('--log_step', default=10, help='path to val images')
opt = parser.parse_args()
opt.cuda = True
print(opt)

os.makedirs(opt.outf, exist_ok=True)
os.makedirs(opt.outf + 'model/', exist_ok=True)
os.makedirs(opt.outf + 'val/', exist_ok=True)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

###########   DATASET   ###########

train_dataset = dataset(opt.dataPath, opt.loadSize_w, opt.loadSize_h, opt.flip)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batchSize, shuffle=True,
                                           num_workers=opt.num_workers)

val_dataset = val_dataset(opt.val_path, opt.loadSize_w, opt.loadSize_h, opt.flip)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=opt.batchSize, shuffle=True,
                                               num_workers=opt.num_workers)


###########   INIT MODEL   ###########
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)


netD = Discriminator(opt.nc, opt.nc, opt.ndf)
netG = Rectifier()

if opt.netG != '' and opt.netD != '':
    netG.load_state_dict(torch.load(opt.netG))
    netD.load_state_dict(torch.load(opt.netD))

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netG = nn.DataParallel(netG, device_ids=[0])
    netD = nn.DataParallel(netD, device_ids=[0])

if opt.netG == '':
    netG.apply(weights_init)
    netD.apply(weights_init)

###########   LOSS & OPTIMIZER   ##########
criterion = nn.BCELoss()
criterionL1 = nn.L1Loss()
criterionS = SampleLoss(min_pixel=-1, max_pixel=1)

optimizerG = torch.optim.Adam(netG.parameters(), lr=0.01, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.SGD(netD.parameters(), lr=0.0001, momentum=0.9)

###########   GLOBAL VARIABLES   ###########
input_nc = opt.nc
output_nc = opt.nc
size = opt.loadSize_w

real_A = torch.FloatTensor(opt.batchSize, input_nc, size, size)
real_A_small = torch.FloatTensor(opt.batchSize, input_nc, size, size)
real_B = torch.FloatTensor(opt.batchSize, input_nc, size, size)
label = torch.FloatTensor(opt.batchSize)

real_A = Variable(real_A)
real_A_small = Variable(real_A_small)
real_B = Variable(real_B)
label = Variable(label)

if opt.cuda:
    real_A = real_A.cuda()
    real_A_small = real_A_small.cuda()
    real_B = real_B.cuda()
    label = label.cuda()

real_label = 1
fake_label = 0


if __name__ == '__main__':
    ########### Training   ###########
    log = open('log.txt', 'w')
    start = time.time()
    netD.train()
    netG.train()
    for epoch in range(1, opt.nepoch + 1):
        loader = iter(train_loader)
        for i in range(0, train_dataset.__len__(), opt.batchSize):
            train_start = time.time()
            ########### fDx ###########
            netD.zero_grad()

            img_distorted_small, img_distorted, img_corrected, name = loader.next()

            real_A_small.resize_(img_distorted_small.size()).copy_(img_distorted_small)
            real_A.resize_(img_distorted.size()).copy_(img_distorted)
            real_B.resize_(img_corrected.size()).copy_(img_corrected)
            real_AB = torch.cat((real_A, real_B), 1)

            output = netD(real_AB)
            label.resize_(output.size())
            label.fill_(real_label)
            errD_real = criterion(output, label)

            errD_real.backward()

            fake_B = netG(real_A, real_A_small)
            label.fill_(fake_label)

            fake_AB = torch.cat((real_A, fake_B), 1)
            output = netD(fake_AB.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()

            errD = (errD_fake + errD_real)/2
            optimizerD.step()

            ########### fGx ###########
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_AB)
            errGAN = criterion(output, label)

            errL1 = criterionL1(fake_B, real_B)
            errS = criterionS(fake_B, real_A)

            errTV = tv_loss(fake_B)
            errG = errGAN + 0.9 * opt.lamb*errL1 + 0.1 * errTV + 0.4 * errS

            errG.backward()
            optimizerG.step()
            train_end = time.time()

            ########### Logging ##########

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_L1: %.4f Loss_TV: %.4f Loss_S: %.4f '
                  'Loss_all: %.4f Time: %.4f'
                  % (epoch, opt.nepoch, i, len(train_loader) * opt.batchSize,
                     errD.item(), errGAN.item(), 0.9*errL1.item(), 0.1*errTV.item(), 0.4 * errS.item(),
                     errG.item(), train_end - train_start))
            log.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_L1: %.4f Loss_TV: %.4f Loss_S: %.4f '
                      'Loss_all: %.4f\n'
                      % (epoch, opt.nepoch, i, len(train_loader) * opt.batchSize,
                         errD.item(), errGAN.item(), 0.9*errL1.item(), 0.1*errTV.item(), 0.4 * errS.item(),
                         errG.item()))

            if i % opt.test_step == 0:
                vutils.save_image(fake_B.data, opt.outf + 'fake_samples_epoch_%03d_%03d.png' % (epoch, i),
                                  normalize=True)

        if epoch % opt.val_epoch == 0:
            print('Validating...')
            loader = iter(val_loader)
            val_distorted_synth_small, val_distorted_synth, name = loader.next()

            real_A.resize_(val_distorted_synth.size()).copy_(val_distorted_synth)
            real_A_small.resize_(val_distorted_synth_small.size()).copy_(val_distorted_synth_small)
            val_corrected_synth = netG(real_A, real_A_small)
            vutils.save_image(val_corrected_synth.data,
                              opt.outf + 'val/fake_samples_epoch_%03d.png' % epoch, normalize=True)
            print('Validation done.')

        if epoch % opt.save_epoch == 0:
            torch.save(netG.state_dict(), '%s/model/netG_%s.pth' % (opt.outf, str(epoch)))
            torch.save(netD.state_dict(), '%s/model/netD_%s.pth' % (opt.outf, str(epoch)))

    end = time.time()
    torch.save(netG.state_dict(), '%s/model/netG_final.pth' % opt.outf)
    torch.save(netD.state_dict(), '%s/model/netD_final.pth' % opt.outf)
    print('Program processed ', end - start, 's, ', (end - start) / 60, 'min, ', (end - start) / 3600, 'h')
    log.close()
