# This code is copied verbatim from https://github.com/Microsoft/robust-models-transfer.
# The original repo is distributed under the MIT license.

from glob import glob 
from . import constants as cs
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from os.path import join as osj
from PIL import Image
from torchvision.transforms import InterpolationMode

class DTD(Dataset):
    def __init__(self, split="1", train=False, interpolation=InterpolationMode.BILINEAR, path=cs.DTD_PATH,
                 use_data_aug=True, use_imagenet_stats=False):
        super().__init__()
        print("INTERPOLATION MODE", interpolation)
        train_path = osj(path, f"labels/train{split}.txt")
        val_path = osj(path, f"labels/val{split}.txt")
        test_path = osj(path, f"labels/test{split}.txt")
        if train:
            self.ims = open(train_path).readlines() + \
                            open(val_path).readlines()
        else:
            self.ims = open(test_path).readlines()
        
        self.full_ims = [osj(path, "images", x) for x in self.ims]
        
        pth = osj(path, f"labels/classes.txt")
        self.c_to_t = {x.strip(): i for i, x in enumerate(open(pth).readlines())}

        means = (0.5322, 0.4734, 0.4251)
        stds = (0.2642, 0.2530, 0.2613)
        if use_imagenet_stats:
            print("Use ImageNet stats for train and test")
            means = (0.485, 0.456, 0.406)
            stds = (0.229, 0.224, 0.225)
        self.transform = cs.train_transforms(means, stds, interpolation) if (train and use_data_aug) else \
                                         cs.test_transforms(means, stds, interpolation)
        self.labels = [self.c_to_t[x.split("/")[0]] for x in self.ims]

    def __getitem__(self, index):
        im = Image.open(self.full_ims[index].strip())
        im = self.transform(im)
        return im, self.labels[index]

    def __len__(self):
        return len(self.ims)

