"""
Example models to train and prune.
Interface is provided by the get_model function.
"""
import torch
from torch.nn import Module
import functools

from models.module import DeepSparseModuleWrapper
from models.resnet_imagenet import *
from models.resnet_cifar10 import *
from models.mobilenet import *
from models.wide_resnet_cifar import *
from models.external.str.STR_resnet import ResNet50 as resnet50_str
from models.external.rigl.resnet import ResnetTF as resnet50_rigl
from models.external.str.STR_mobilenet import MobileNetV1 as mobilenet_str
from models.external.amc.mobilenet_v1 import MobileNet as mobilenet_amc

from torchvision.models import resnet50 as resnet50_torch
from torchvision.models import vgg16_bn
from torchvision.models import vgg11, vgg19, vgg11_bn

import pdb

CIFAR10_MODELS = ['resnet20', 'resnet32', 'resnet44', 'resnet56']
CIFAR100_MODELS = ['wideresnet', 'resnet20']
IMAGENET_MODELS = ['resnet18', 'resnet34', 'resnet50', 'vgg19', 'resnet101', 'resnet152', 'mobilenet', 'mobilenet_str', 'mobilenet_amc', 
                    'resnet50_str', 'resnet50_rigl', 'resnet50_torch']

DATASET_NUM_CLASSES={
    'imagenet':     1000,
    'aircraft':     100,
    'birds':        500,
    'caltech101':   101,
    'caltech256':   257,
    'cars':         196,
    'cifar10':      10,
    'cifar100':     100,
    'dtd':          47,
    'food101':      101,
    'flowers':      102,
    'pets':         37,
    'SUN':          397,
        }

def model_wrapper_decorator(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs) -> Module:
        deepsparse_wrapper = None
        if "deepsparse_wrapper" in kwargs:
            deepsparse_wrapper = kwargs["deepsparse_wrapper"]
            del kwargs["deepsparse_wrapper"]
        wrapper_input_shape = None
        if "wrapper_input_shape" in kwargs:
            wrapper_input_shape = kwargs["wrapper_input_shape"]
            del kwargs["wrapper_input_shape"]

        mod = func(*args, **kwargs)

        if deepsparse_wrapper:
            mod = DeepSparseModuleWrapper(mod, wrapper_input_shape)

        return mod

    return _wrapper



@model_wrapper_decorator
def get_model(arch, dataset, pretrained=False, args=None):
    if arch.startswith('resnet20') and dataset == 'cifar10':
        try:
            return globals()[arch]()
        except:
            raise ValueError(f'Model {arch} is not supported for {dataset}, list of supported: {", ".join(CIFAR10_MODELS)}')
    if 'resnet' in arch and any([dataset == 'imagenet', dataset == 'imagenette']):
        if dataset == 'imagenette':
            kwargs_dict['num_classes'] = 10
            return globals()[arch](**kwargs_dict)
        return globals()[arch](pretrained)
    if arch=='wideresnet' and dataset=='cifar100':
        model = Wide_ResNet(28, 10, 0.3, 100)
        return model
    if arch=='resnet20' and dataset=='cifar100':
        model = resnet20(num_classes=100)
        return model
    if 'resnet50' in arch and dataset in DATASET_NUM_CLASSES.keys():
        return globals()[arch](pretrained=False, num_classes=DATASET_NUM_CLASSES[dataset])   
    if 'mobilenet' in arch and dataset in DATASET_NUM_CLASSES.keys():
        return globals()[arch](num_classes=DATASET_NUM_CLASSES[dataset]) 
    return globals()[arch](pretrained, num_classes=DATASET_NUM_CLASSES[dataset])   
        
    raise NotImplementedError

if __name__ == '__main__':
    get_model('resnet', 'cifar10', pretrained=False)
