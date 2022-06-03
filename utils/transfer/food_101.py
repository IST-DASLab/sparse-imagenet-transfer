# This code is copied verbatim from https://github.com/Microsoft/robust-models-transfer.
# The original repo is distributed under the MIT license.

# pytorch imports
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from robustness import data_augmentation as da
from . import constants as cs

class FOOD101():
    def __init__(self, path):
        self.TRAIN_PATH = path +"/train"
        self.VALID_PATH = path+"/valid"

        self.train_ds, self.valid_ds, self.train_cls, self.valid_cls = [None]*4
   
    def _get_tfms(self, interpolation, use_imagenet_stats=False):
        means = (0.5566, 0.4378, 0.3193)
        stds = (0.2589, 0.2622, 0.2630)
        if use_imagenet_stats:
            print("Use ImageNet stats for train and test")
            means = (0.485, 0.456, 0.406)
            stds = (0.229, 0.224, 0.225)

        train_tfms = cs.train_transforms(means, stds, interpolation)
        valid_tfms = cs.test_transforms(means, stds, interpolation)
        return train_tfms, valid_tfms            
            
    def get_dataset(self, interpolation, use_data_aug=True, use_imagenet_stats=False):
        train_tfms, valid_tfms = self._get_tfms(interpolation, use_imagenet_stats=use_imagenet_stats) # transformations
        if use_data_aug:
            self.train_ds = datasets.ImageFolder(root=self.TRAIN_PATH, transform=train_tfms)
        else:
            self.train_ds = datasets.ImageFolder(root=self.TRAIN_PATH, transform=valid_tfms)
        self.valid_ds = datasets.ImageFolder(root=self.VALID_PATH,
                                        transform=valid_tfms)        
        self.train_classes = self.train_ds.classes
        self.valid_classes = self.valid_ds.classes

        assert self.train_classes==self.valid_classes
        return self.train_ds, self.valid_ds, self.train_classes
    
    def get_dls(self, train_ds, valid_ds, bs, **kwargs):
        return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
               DataLoader(valid_ds, batch_size=bs, shuffle=True, **kwargs))
   
