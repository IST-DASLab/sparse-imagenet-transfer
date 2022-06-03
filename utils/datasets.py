"""
Dataset loading utilities
"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode
import sklearn.datasets as sklearn_datasets

from torch.utils.data import TensorDataset, Subset

from utils.transfer.caltech import Caltech101, Caltech256
from utils.transfer.food_101 import FOOD101
from utils.transfer.dtd import DTD
from utils.transfer.aircraft import FGVCAircraft
from utils.transfer import constants as cs
from utils.transfer.transfer_datasets import TransformedDataset
from PIL import Image

import pdb

DATASETS_NAMES = ['imagenet', 'cifar10', 'cifar100', 'mnist', 'food101',
                  'caltech101', 'caltech256', 'pets', 'birds', 'flowers', 'celeba', 'awa2']

__all__ = ["get_datasets", "extract_resnet50_features", "extract_mobilenet_features", "classification_num_classes", "interpolation_flag"]



def classification_dataset_str_from_arch(arch):
    if 'cifar100' in arch:
        dataset = 'cifar100'
    elif 'cifar' in arch:
        dataset = 'cifar10'
    elif 'mnist' in arch:
        dataset = 'mnist'
    else:
        dataset = 'imagenet'
    return dataset


def extract_resnet50_features(model, data_loader, device, update_bn_stats=False, bn_updates=10):
    
    # this is where the extracted features will be added:
    data_features_tensor = None
    data_labels_tensor = None

    
    if update_bn_stats:
        # if true, update the BatchNorm stats by doing a few dummy forward passes through the data
        # if false, use the ImageNet Batch Norm stats
        model.train()
        for i in range(bn_updates):
            with torch.no_grad():
                for sample, target in data_loader:
                    sample = sample.to(device)
                    model(sample)
    model.eval()

    #register a forward hook: 
    h = None
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    h = model.avgpool.register_forward_hook(get_activation('avgpool'))

    # get the features before the FC layer
    with torch.no_grad():
        for i, (sample, target) in enumerate(data_loader):
            sample = sample.to(device)
            sample_output = model(sample)
            sample_feature = activation['avgpool'].cpu()
            
            sample_feature = sample_feature.view(sample_feature.size(0), -1)
            if data_features_tensor is None:
                data_features_tensor = sample_feature
                data_labels_tensor = target
            else:
                data_features_tensor = torch.cat((data_features_tensor, sample_feature))
                data_labels_tensor = torch.cat((data_labels_tensor, target))
            if i % 100==0:
                print("extracted for {} batches".format(i))
    h.remove()
    #tensor_features_data = torch.utils.data.TensorDataset(data_features_tensor, data_labels_tensor)
    return data_features_tensor, data_labels_tensor


def extract_mobilenet_features(model, data_loader, device, amc=False, update_bn_stats=False, bn_updates=10):
    
    # this is where the extracted features will be added:
    data_features_tensor = None
    data_labels_tensor = None

    if update_bn_stats:
        # if true, update the BatchNorm stats by doing a few dummy forward passes through the data
        # if false, use the ImageNet Batch Norm stats
        model.train()
        for i in range(bn_updates):
            with torch.no_grad():
                for sample, target in data_loader:
                    sample = sample.to(device)
                    model(sample)
    model.eval()

    #register a forward hook:
    h = None
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    if amc:
        h = model.features.register_forward_hook(get_activation('features'))
    else:
        h = model.model.register_forward_hook(get_activation('model'))

    # get the features before the FC layer
    with torch.no_grad():
        for i, (sample, target) in enumerate(data_loader):
            sample = sample.to(device)
            sample_output = model(sample)
            if amc:
                sample_feature = activation['features'].cpu()
                sample_feature = sample_feature.mean(3).mean(2)
            else:
                sample_feature = activation['model'].cpu()
                sample_feature = sample_feature.view(sample_feature.size(0), -1)

            if data_features_tensor is None:
                data_features_tensor = sample_feature
                data_labels_tensor = target
            else:
                data_features_tensor = torch.cat((data_features_tensor, sample_feature))
                data_labels_tensor = torch.cat((data_labels_tensor, target))
            if i % 100==0:
                print("extracted for {} batches".format(i))
    h.remove()
    return data_features_tensor, data_labels_tensor



def classification_num_classes(dataset):
    return {'cifar10': 10,
            'cifar100': 100,
            'mnist': 10,
            'imagenet': 1000,
            'food101': 101,
            'caltech101': 101,
            'caltech256': 257,
            'pets': 37,
            'birds': 500,
            'flowers': 102,
            'SUN': 397,
            'cars': 196,
            'dtd': 47,
            'aircraft': 100,
            'celeba': 40,
            'awa2': 50,
            }.get(dataset, None)


def classification_get_input_shape(dataset):
    if dataset=='imagenet':
        return 1, 3, 224, 224
    elif dataset in ('cifar10', 'cifar100'):
        return 1, 3, 32, 32
    elif dataset == 'mnist':
        return 1, 1, 28, 28
    elif dataset=='food101':
        return 1, 3, 224, 224
    elif dataset=='caltech101':  # Actually these don't have a consistent size...
        return 1, 3, 224, 224
    elif dataset=='caltech256':
        return 1, 3, 224, 224
    elif dataset=='pets':
        return 1, 3, 224, 224
    elif dataset=='birds':
        return 1, 3, 224, 224
    elif dataset=='flowers':
        return 1, 3, 224, 224
    elif dataset=='SUN':
        return 1, 3, 224, 224
    elif dataset=='cars':
        return 1, 3, 224, 224
    elif dataset=='dtd':
        return 1, 3, 224, 224
    elif dataset=='aircraft':
        return 1, 3, 224, 224
    elif dataset=='celeba':
        return 1, 3, 224, 224
    elif dataset=='awa2':
        return 1, 3, 224, 224
    else:
        raise ValueError("dataset %s is not supported" % dataset)


def __dataset_factory(dataset):
    return globals()[f'{dataset}_get_datasets']

def interpolation_flag(interpolation):
    if interpolation == 'bilinear':
        return InterpolationMode.BILINEAR
    elif interpolation == 'bicubic':
        return InterpolationMode.BICUBIC
    raise ValueError("interpolation must be one of 'bilinear', 'bicubic'")

def get_datasets(dataset, dataset_dir, **kwargs):
    datasets_fn = __dataset_factory(dataset)
    # if dataset == 'imagenet':
    #     return datasets_fn(dataset_dir, kwargs['use_aa'],
    #         kwargs['use_ra'], kwargs['remode'], kwargs['reprob'], kwargs['num_aug_splits'], kwargs['use_data_aug'])
    return datasets_fn(dataset_dir, **kwargs)

def blobs_get_datasets(dataset_dir=None):
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    if os.path.isdir(dataset_dir):
        X_train, Y_train = torch.load(os.path.join(train_dir, 'x_data.pth')),\
                           torch.load(os.path.join(train_dir, 'y_data.pth'))
        X_test, Y_test = torch.load(os.path.join(test_dir, 'x_data.pth')),\
                         torch.load(os.path.join(test_dir, 'y_data.pth'))
    else:
        X, Y = sklearn_datasets.make_blobs(n_samples=15000,
                                           n_features=5,
                                           centers=3)
        X_train, Y_train = torch.FloatTensor(X[:-5000]), torch.FloatTensor(Y[:-5000])
        X_test, Y_test = torch.FloatTensor(X[-5000:]), torch.FloatTensor(Y[-5000:])
        
        # making dirs to save train/test
        os.makedirs(train_dir)
        os.makedirs(test_dir)

        torch.save(X_train, os.path.join(train_dir, 'x_data.pth'))
        torch.save(Y_train, os.path.join(train_dir, 'y_data.pth'))
        torch.save(X_test, os.path.join(test_dir, 'x_data.pth'))
        torch.save(Y_test, os.path.join(test_dir, 'y_data.pth'))

    # making torch datasets
    train_dataset = TensorDataset(X_train, Y_train.long())
    test_dataset = TensorDataset(X_test, Y_test.long())

    return train_dataset, test_dataset

def mnist_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear'):
    # interpolation not used, here for consistent call.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True,
                                   download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root=data_dir, train=False,
                                  transform=test_transform)

    return train_dataset, test_dataset


def cifar10_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2023, 0.1994, 0.2010) 
    
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)

    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),#transforms.RandomCrop(32, padding=4),transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation), 
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)


    test_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means, stds)
                                        ])

    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def cifar100_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    
    means = (0.5071, 0.4867, 0.4408)
    stds = (0.2675, 0.2565, 0.2761)
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)

    
    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(), 
                                         transforms.Normalize(means, stds)])

    test_dataset = datasets.CIFAR100(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def food101_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    food = FOOD101(data_dir)
    train_ds, test_ds, num_classes = food.get_dataset(use_data_aug=use_data_aug, interpolation=interpolation,
                                                      use_imagenet_stats=use_imagenet_stats)
    return train_ds, test_ds


def dtd_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    return DTD(train=True, path=data_dir, use_data_aug=use_data_aug, interpolation=interpolation, use_imagenet_stats=use_imagenet_stats), \
           DTD(train=False, path=data_dir, interpolation=interpolation, use_imagenet_stats=use_imagenet_stats)


def caltech101_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    ds = Caltech101(data_dir, download=True)
    NUM_TRAINING_SAMPLES_PER_CLASS=30
    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    means = (0.5413, 0.5063, 0.4693)
    stds = (0.3115, 0.3090, 0.3183)
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
        
    if use_data_aug:
        train_set = TransformedDataset(train_set, transform=cs.train_transforms(means, stds, interpolation)) 
    else:
        print("Use ImageNet stats for train")
        train_set = TransformedDataset(train_set, transform=cs.test_transforms(means, stds, interpolation)) 
    
    
    test_set = TransformedDataset(test_set,
            transform=cs.test_transforms(means, stds, interpolation))
    return train_set, test_set


def caltech256_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    ds = Caltech256(data_dir, download=True)
    NUM_TRAINING_SAMPLES_PER_CLASS=60
    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    means = (0.5438, 0.5141, 0.4821)
    stds = (0.3077, 0.3044, 0.3163)
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
    
    if use_data_aug:
        train_set = TransformedDataset(train_set, transform=cs.train_transforms(means, stds, interpolation)) 
    else:
        print("Use ImageNet stats for train")
        train_set = TransformedDataset(train_set, transform=cs.test_transforms(means, stds, interpolation)) 
    
    
    test_set = TransformedDataset(test_set, transform=cs.test_transforms(means, stds, interpolation))
    return train_set, test_set


def aircraft_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    means = (0.4892, 0.5159, 0.5356)
    stds = (0.2275, 0.2200, 0.2476)
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
    
    if use_data_aug:
        train_ds = FGVCAircraft(root=data_dir, train=True, download=True, transform=cs.train_transforms(means, stds, interpolation))
    else:
        print("Use ImageNet stats for train")
        train_ds = FGVCAircraft(root=data_dir, train=True, download=True, transform=cs.test_transforms(means, stds, interpolation))
    
    test_ds = FGVCAircraft(root=data_dir, train=False, download=True, transform=cs.test_transforms(means, stds, interpolation))
    return train_ds, test_ds


def birds_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    means = (0.4856, 0.4948, 0.4486)
    stds = (0.2233, 0.2213, 0.2586)
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)

    
    train_path = os.path.join(data_dir, "train")
    if use_data_aug:
        train_ds = datasets.ImageFolder(root=train_path, transform=cs.train_transforms(means, stds, interpolation))
    else:
        print("Use ImageNet stats")
        train_ds = datasets.ImageFolder(root=train_path, transform=cs.test_transforms(means, stds, interpolation))
    
   
    test_path = os.path.join(data_dir, "test")
    test_ds = datasets.ImageFolder(root=test_path, transform=cs.test_transforms(means, stds, interpolation))
    return train_ds, test_ds


def pets_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    means = (0.4807, 0.4432, 0.3949)
    stds = (0.2598, 0.2537, 0.2597)
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)

    train_path = os.path.join(data_dir, "train")
    if use_data_aug:
        train_ds = datasets.ImageFolder(root=train_path, transform=cs.train_transforms(means, stds, interpolation))
    else:
        train_ds = datasets.ImageFolder(root=train_path, transform=cs.test_transforms(means, stds, interpolation))
      
    test_path = os.path.join(data_dir, "test")
    test_ds = datasets.ImageFolder(root=test_path, transform=cs.test_transforms(means, stds, interpolation))
    return train_ds, test_ds


def flowers_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    means = (0.5133, 0.4148, 0.3383)
    stds = (0.2959, 0.2502, 0.2900)
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
     
    train_path = os.path.join(data_dir, "train")
    if use_data_aug:
        train_ds = datasets.ImageFolder(root=train_path, transform=cs.train_transforms(means, stds, interpolation))
    else:
        print("Use ImageNet stats for train")
        train_ds = datasets.ImageFolder(root=train_path, transform=cs.test_transforms(means, stds, interpolation)) 
   
    test_path = os.path.join(data_dir, "test")
    test_ds = datasets.ImageFolder(root=test_path, transform=cs.test_transforms(means, stds, interpolation))
    return train_ds, test_ds


def SUN_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    means = (0.4772, 0.4566, 0.4168)
    stds = (0.2556, 0.2520, 0.2720)
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
        
    train_path = os.path.join(data_dir, "train")
    if use_data_aug:
        train_ds = datasets.ImageFolder(root=train_path, transform=cs.train_transforms(means, stds, interpolation))
    else:
        print("Use ImageNet stats for train")
        train_ds = datasets.ImageFolder(root=train_path, transform=cs.test_transforms(means, stds, interpolation)) 

    test_path = os.path.join(data_dir, "test")
    test_ds = datasets.ImageFolder(root=test_path, transform=cs.test_transforms(means, stds, interpolation))
    return train_ds, test_ds


def cars_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    means = (0.4513, 0.4354, 0.4358)
    stds = (0.2900, 0.2880, 0.2951)
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225) 
    
    train_path = os.path.join(data_dir, "train")
    if use_data_aug:
        train_ds = datasets.ImageFolder(root=train_path, transform=cs.train_transforms(means, stds, interpolation))
    else:
        print("Use ImageNet stats for train")
        train_ds = datasets.ImageFolder(root=train_path, transform=cs.test_transforms(means, stds, interpolation)) 
    
    test_path = os.path.join(data_dir, "test")
    test_ds = datasets.ImageFolder(root=test_path, transform=cs.test_transforms(means, stds, interpolation))
    return train_ds, test_ds


def imagenet_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear'):
    interpolation = interpolation_flag(interpolation)
    print("getting imagenet datasets")
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation = interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)


    return train_dataset, test_dataset



def celeba_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear'):
    interpolation = interpolation_flag(interpolation)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(224, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    train_dataset = datasets.CelebA(root=data_dir, split='train',
            target_type='attr', transform=train_transform, download=True)

    test_transform = transforms.Compose([
        transforms.Resize(224, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.CelebA(root=data_dir, split='test',
            target_type='attr', transform=test_transform, download=True)



    return train_dataset, test_dataset



def awa2_get_datasets(data_dir, use_data_aug=True, split_size=0.8, split_seed=0, interpolation='bilinear'):
    interpolation = interpolation_flag(interpolation)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ])
    

    full_dataset = datasets.ImageFolder(data_dir, train_transform)
    full_dataset_no_da = datasets.ImageFolder(data_dir, test_transform)
    total_n_samples = len(full_dataset)
    np.random.seed(split_seed)
    perm_idxs = np.random.permutation(total_n_samples)
    trn_size = int(split_size * total_n_samples)
    trn_idxs = perm_idxs[:trn_size]
    tst_idxs = perm_idxs[trn_size:]
    train_dataset = torch.utils.data.Subset(full_dataset, trn_idxs)
    test_dataset = torch.utils.data.Subset(full_dataset_no_da, tst_idxs)
    return train_dataset, test_dataset



