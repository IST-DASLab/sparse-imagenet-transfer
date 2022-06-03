"""
This module implements training policies.
For most usecases, only one trainer instance is needed for training and pruning
with a single model. Several trainers can be used for training with knowledge distillation.
"""

import numpy as np
import torch
import torch.nn as nn
from optimization.sgd import SGD
# from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.cuda.amp import autocast
import torch.nn.functional as F
import logging
# import torchcontrib

from policies.policy import PolicyBase
from optimization.lr_schedulers import StageExponentialLR, CosineLR




def build_optimizer_from_config(model, optimizer_config, update_modules=None):
    optimizer_class = optimizer_config['class']
    restricted_keys = ['class', 'modules']
    optimizer_args = {k: v for k, v in optimizer_config.items() if k not in restricted_keys}
    if update_modules is None:
        update_params = model.parameters()
    else:
        update_params = [p for n, p in model.named_parameters() if n[7:] in update_modules]
        print("The following will be updated: ", len(update_params))
    optimizer_args['params'] = model.parameters()
    optimizer = globals()[optimizer_class](**optimizer_args)
    return optimizer


def build_lr_scheduler_from_config(optimizer, lr_scheduler_config):
    lr_scheduler_class = lr_scheduler_config['class']
    lr_scheduler_args = {k: v for k, v in lr_scheduler_config.items() if k != 'class'}
    lr_scheduler_args['optimizer'] = optimizer
    epochs = lr_scheduler_args['epochs']
    lr_scheduler_args.pop('epochs')
    lr_scheduler = globals()[lr_scheduler_class](**lr_scheduler_args)
    return lr_scheduler, epochs


def build_training_policy_from_config(model, scheduler_dict, trainer_name,
                                      fp16_scaler=None, update_modules=None):
    print("Update modules at going to trainer", update_modules)
    trainer_dict = scheduler_dict['trainers'][trainer_name]
    optimizer = build_optimizer_from_config(model, trainer_dict['optimizer'], update_modules)
    lr_scheduler, epochs = build_lr_scheduler_from_config(optimizer, trainer_dict['lr_scheduler'])
    return TrainingPolicy(model, optimizer, lr_scheduler, epochs, fp16_scaler=fp16_scaler)


class TrainingPolicy(PolicyBase):
    def __init__(self, model, optimizer, lr_scheduler, epochs, fp16_scaler=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.model = model
        self.fp16_scaler = fp16_scaler
        self.enable_autocast = False
        if fp16_scaler is not None:
            self.enable_autocast = True
        print("initial optim lr", self.optim_lr)
        self.loss = F.cross_entropy

    def eval_model(self, loader, device, epoch_num, avg_per_class=False):
        self.model.eval()
        eval_loss = 0
        total_correct = 0
        per_class_correct = None
        per_class_counts = None
        with torch.no_grad():
            for in_tensor, target in loader:
                in_tensor, target = in_tensor.to(device), target.to(device)
                output = self.model(in_tensor)
                eval_loss += self.loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred))
                total_correct += correct.sum().item()
                if avg_per_class:
                    per_class_pred = F.one_hot(output.argmax(dim=1), num_classes = output.shape[1])
                    if per_class_correct == None:
                        per_class_correct = (per_class_pred * correct).sum(dim=0)
                    else:
                        per_class_correct+= (per_class_pred * correct).sum(dim=0)
                    if per_class_counts == None:
                        per_class_counts = F.one_hot(target, num_classes = output.shape[1]).sum(dim=0)
                    else:
                        per_class_counts += F.one_hot(target, num_classes = output.shape[1]).sum(dim=0)
        avg_per_class_accuracy = None
        if avg_per_class:
            avg_per_class_accuracy = (per_class_correct / per_class_counts.to(device)).mean().item()
        eval_loss /= len(loader.dataset)
        return eval_loss, total_correct, avg_per_class_accuracy

    @property
    def optim_lr(self):
        return list(self.optimizer.param_groups)[0]['lr']


    def on_minibatch_begin(self, minibatch, device, **kwargs):
        """
        Loss can be composite, e.g., if we want to add some KD or
        regularization in future
        """
        self.model.train()
        self.optimizer.zero_grad()
        in_tensor, target = minibatch
        in_tensor, target = in_tensor.to(device), target.to(device)
        with autocast(enabled=self.enable_autocast):
            output = self.model(in_tensor)
            loss = self.loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = 1.0 * correct / target.size(0)
        loss = torch.sum(loss)
        acc = np.sum(acc)
        return loss, acc

    def on_parameter_optimization(self, loss, epoch_num, **kwargs):
        if self.enable_autocast:
            self.fp16_scaler.scale(loss).backward()
            self.fp16_scaler.step(self.optimizer)
            self.fp16_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


    def on_epoch_end(self, device, epoch_num, **kwargs):
        start, freq, end = self.epochs
        if (epoch_num - start) % freq == 0 and epoch_num < end + 1 and start - 1 < epoch_num:
            self.lr_scheduler.step()
        


if __name__ == '__main__':
    """
    TODO: remove after debug
    """
    from efficientnet_pytorch import EfficientNet
    from masking_utils import get_wrapped_model

    from utils import read_config
    path = "./configs/test_config.yaml"
    sched_dict = read_config(stream)

    model = get_wrapped_model(EfficientNet.from_pretrained('efficientnet-b1'))
    optimizer = build_optimizer_from_config(model, sched_dict['optimizer'])
    lr_scheduler,_ = build_lr_scheduler_from_config(optimizer, sched_dict['lr_scheduler'])
    training_policy = build_training_policy_from_config(model, sched_dict)

