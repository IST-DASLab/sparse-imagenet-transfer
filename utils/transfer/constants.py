# This code is copied verbatim from https://github.com/Microsoft/robust-models-transfer.
# The original repo is distributed under the MIT license.

from torchvision import transforms

# Planes dataset
FGVC_PATH = "/tmp/datasets/fgvc-aircraft-2013b/"

# Oxford Flowers dataset
FLOWERS_PATH = "/tmp/datasets/oxford_flowers_pytorch/"

# DTD dataset
DTD_PATH="/tmp/datasets/dtd/"

# Stanford Cars dataset
CARS_PATH = "/tmp/datasets/cars_new"

# SUN397 dataset
SUN_PATH="/tmp/datasets/SUN397/splits_01/"

# FOOD dataset
FOOD_PATH = "/tmp/food-101"

# BIRDS dataset
BIRDS_PATH = "/tmp/datasets/birdsnap"

# PETS dataset
PETS_PATH = "/tmp/datasets/pets"

# Caltech datasets
CALTECH101_PATH = "/tmp/datasets"
CALTECH256_PATH = "/tmp/datasets"

# Data Augmentation defaults
def train_transforms(means, stds, interpolation):
    return transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224, interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])

def test_transforms(means, stds, interpolation):
    return transforms.Compose([
        # transforms.Resize(32),
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])
