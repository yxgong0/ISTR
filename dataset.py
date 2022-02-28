import torch
import numpy as np
import torch.utils.data as data
from os import listdir
import os
from PIL import Image
import random
# from util.image_rotation import Rotator
import six
import sys
import lmdb
import torchvision.transforms as transforms

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
sets = ['ICDAR', 'SYNTH', 'SYNTHDATA']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def default_loader(path):
    return Image.open(path).convert('L')


def ToTensor(pic):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backard compability
        return img.float().div(255)
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
        # return img
    else:
        return img

# You should build custom dataset as below.
class dataset(data.Dataset):
    def __init__(self,dataPath='',size_w=128, size_h=64,flip=0):
        super(dataset, self).__init__()
        self.list = [x for x in listdir(dataPath + 'distorted/') if is_image_file(x)]
        self.dataPath = dataPath
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip

    def __getitem__(self, index):
        path = os.path.join(self.dataPath + 'distorted/',self.list[index])
        path2 = os.path.join(self.dataPath + 'corrected/', self.list[index])
        try:
            img = Image.open(path).convert('L')
            img2 = Image.open(path2).convert('L')

        except OSError:
            return None, None, None, None

        size_h = self.size_h
        size_w = self.size_w

        img0 = img.resize((int(size_w/2), int(size_h/2)), Image.BILINEAR)
        img = img.resize((size_w, size_h), Image.BILINEAR)

        img2 = img2.resize((size_w, size_h), Image.BILINEAR)

        if self.flip == 1:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

                img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)

                img0 = img0.transpose(Image.FLIP_LEFT_RIGHT)

        img = ToTensor(img)
        img = img.mul_(2).add_(-1)
        img2 = ToTensor(img2)
        img2 = img2.mul_(2).add_(-1)
        img0 = ToTensor(img0)
        img0 = img0.mul_(2).add_(-1)

        return img0, img, img2, self.list[index]

    def __len__(self):
        return len(self.list)


class val_dataset(data.Dataset):
    def __init__(self,dataPath='',size_w=128,size_h=64,flip=0):
        super(val_dataset, self).__init__()
        self.list = [x for x in listdir(dataPath) if is_image_file(x)]
        self.dataPath = dataPath
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip

        self.checklist()

    def __getitem__(self, index):
        path = os.path.join(self.dataPath,self.list[index])
        img = default_loader(path)
        size_h = self.size_h
        size_w = self.size_w
        img0 = img.resize((int(size_w/2), int(size_h/2)), Image.BILINEAR)
        img = img.resize((size_w, size_h), Image.BILINEAR)

        if self.flip == 1:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img0 = img0.transpose(Image.FLIP_LEFT_RIGHT)
        img = ToTensor(img)
        img = img.mul_(2).add_(-1)
        img0 = ToTensor(img0)
        img0 = img0.mul_(2).add_(-1)

        return img0, img, self.list[index]

    def __len__(self):
        return len(self.list)

    def checklist(self):
        new_list = []
        for index in range(self.list.__len__()):
            path = os.path.join(self.dataPath, self.list[index])
            img = Image.open(path)
            if img is None:
                continue
            new_list.append(self.list[index])
        self.list = new_list


class ResizeNormalize(object):

    def __init__(self, size, colored=False, interpolation=Image.BILINEAR):
        self.size = size
        self.colored = colored
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        if not self.colored:
            img.sub_(0.5).div_(0.5)
        else:
            pass
        return img


class AlignCollate(object):

    def __init__(self, im_h=32, im_w=100, keep_ratio=False, min_ratio=1):
        self.im_h = im_h
        self.im_w = im_w
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        im_h = self.im_h
        im_w = self.im_w
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            im_w = int(np.floor(max_ratio * im_h))
            im_w = max(im_h * self.min_ratio, im_w)

        transform = ResizeNormalize(size=(im_w, im_h))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
