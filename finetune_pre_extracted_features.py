import sys
import os
import time
import argparse
import collections
import numpy as np
import scipy.io
import logging 

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from models import get_model
from utils import (read_config,
                   get_datasets,
                   save_checkpoint,
                   classification_num_classes,
                   get_wrapped_model,
                   preprocess_for_device,
                   load_checkpoint, 
                   extract_resnet50_features,
                   extract_mobilenet_features,
                   TrainingProgressTracker)
from policies.trainers import build_training_policy_from_config

import wandb

import pdb

def setup_logging(args):

    from importlib import reload
    reload(logging)

    # attrs independent of checkpoint restore
    args.logging_level = getattr(logging, args.logging_level.upper())
    upstream_ckpt_name = os.path.basename(args.from_checkpoint_path).split('.')[0]

    
    # get the folders for saved checkpoints
    args.experiment_root_path = 'experiments_transfer'
    args.exp_name = "{:}-{:}-preextracted".format(upstream_ckpt_name, args.dataset)
    args.exp_dir = os.path.join(args.experiment_root_path, args.exp_name)
    run_id = time.strftime('%Y%m%d%H%M%S')
    args.run_dir = os.path.join(args.exp_dir, os.path.join('seed{:}'.format(args.seed), run_id))
    os.makedirs(args.run_dir, exist_ok=True)



    # Make directories
    #os.makedirs(args.run_dir, exist_ok=True)

    log_file_path = os.path.join(args.run_dir, 'log')
    # in append mode, because we may want to restore checkpoints
    logging.basicConfig(filename=log_file_path, filemode='a',
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=args.logging_level)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    return args


def get_parser():
    parser = argparse.ArgumentParser(description='Linear finetuning with pre-extracted features (uses SGD optmizer)')
    parser.add_argument('--cpu', action='store_true', help='force training on CPU')
    parser.add_argument('--gpus', default=None, 
                        help='Comma-separated list of GPU device ids to use, this assumes that parallel is applied (default: all devices)')
    parser.add_argument('--from_checkpoint_path', type=str, default=None, help='sparsity model type')
    parser.add_argument('--dataset', type=str, default='dtd', help='dataset to finetune on')
    parser.add_argument('--dataset_path', type=str, default=None, help='path to the finetuning dataset')
    parser.add_argument('--training_config_path', type=str, default=None, help='path to the yaml file for the training config values')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size used during finetuning')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--logging_level', type=str, default='info', help='logging level: debug, info, warning, error, critical (default: info)')
    parser.add_argument('--test_chkpt', action='store_true', help='whether or not to test if chkpt loaded properly')
    return parser.parse_args()


args = get_parser()
args = setup_logging(args)
args = preprocess_for_device(args)

# get the config with training hyperparams
args.training_config = read_config(args.training_config_path)


# set the seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#-------------------------------------------
# Get the checkpoint path and load the model
#-------------------------------------------


# get the correct checkpoint loader

#----------------
# Get the model
#----------------
# Special case: if the checkpoint_path is set to "torchvision", then
# instead of loading from a checkpoint we use the built-in pretrained
# model.
# TODO: do we need this?
if args.from_checkpoint_path=='torchvision':
    imgnet_model = torchvision.models.resnet50(pretrained=True)
    arch = "resnet50"
    base_arch = "resnet50"
else:
    arch, imgnet_model = load_checkpoint(args.from_checkpoint_path)
    base_arch = None
    for ba in ("resnet50", "resnet34", "resnet18", "mobilenet"):
        if arch.startswith(ba):
            base_arch = ba
    if not base_arch:
        raise ValueError(f"Architecture {arch} does not seem to match any of resnet18 | resnet34 | resnet50 | mobilenet")

imgnet_model.to(args.device)

#-----------------------------------------
# Test the checkpoint loader (optional)
#-----------------------------------------
def test_imgnet_model(model, device, data_loader):
    model.eval()
    data_loss = 0.
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion.to(device)
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            pred = logits.argmax(dim=1, keepdim=True)
            data_loss += criterion(logits, target).item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    data_loss /= len(data_loader.dataset)
    acc_model = 100. * correct / len(data_loader.dataset)
    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(data_loss, correct,
                                                                                len(data_loader.dataset),
                                                                                acc_model))
    return acc_model

imagenet_path = os.environ.get('IMAGENET_PATH')
if args.test_chkpt:
    imgnet_train, imgnet_test = get_datasets('imagenet', imagenet_path, use_data_aug=True)
    imgnet_test_loader = DataLoader(imgnet_test, batch_size=128, shuffle=False, num_workers=10)
    acc_imgnet_val = test_imgnet_model(imgnet_model, args.device, imgnet_test_loader)
    print(f"Imagenet validation accuracy: {acc_imgnet_val}")

    print("-------Test sparsity-------------")
    total_params = 0.
    total_zeros = 0.
    for name, w in imgnet_model.named_parameters():
        total_params_layer = w.data.numel()
        total_zeros_layer = (w.data == 0.).float().sum()
        print(name, total_zeros_layer/total_params_layer)
        total_params += total_params_layer
        total_zeros += total_zeros_layer
    print(f"total sparsity:{total_zeros / total_params}") 
    sys.exit()

#-------------------------------
# Get the pre-extracted data
#-------------------------------

# Define the datasets
args.interpolation = 'bilinear'
if 'rigl' in arch:
    args.interpolation = 'bicubic'
print(f"Using {args.interpolation} interpolation.")
data_train, data_test = get_datasets(args.dataset, args.dataset_path, use_data_aug=False, interpolation=args.interpolation,
                                     use_imagenet_stats=True)
num_classes = classification_num_classes(args.dataset)
train_loader = DataLoader(data_train, batch_size=128, shuffle=False, num_workers=4)
test_loader = DataLoader(data_test, batch_size=128, shuffle=False, num_workers=4)

# feature extraction
t = time.time()
if base_arch.startswith('resnet'):
    trn_features, trn_labels = extract_resnet50_features(imgnet_model, train_loader, device=args.device)
    logging.info("Done extracting train features in {:.2f}s!".format(time.time() - t))
    t - time.time()
    tst_features, tst_labels = extract_resnet50_features(imgnet_model, test_loader, device=args.device)
else:
    amc = False
    if arch=='mobilenet_amc':
        amc = True
    trn_features, trn_labels = extract_mobilenet_features(imgnet_model, train_loader, device=args.device, amc=amc)
    logging.info("Done extracting train features in {:.2f}s!".format(time.time() - t))
    t - time.time()
    tst_features, tst_labels = extract_mobilenet_features(imgnet_model, test_loader, device=args.device, amc=amc)
logging.info("Done extracting test features in {:.2f}s!".format(time.time() - t))
# get the corresponding data loaders for extracted features
train_features_data = torch.utils.data.TensorDataset(trn_features, trn_labels)
train_features_loader = DataLoader(train_features_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_features_data = torch.utils.data.TensorDataset(tst_features, tst_labels)
test_features_loader = DataLoader(test_features_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

#--------------------------
# Set the wandb stats
#--------------------------
config_dictionary = dict(yaml=args.training_config_path, params=args, ckpt_path=args.run_dir)
upstream_ckpt_name = os.path.basename(args.from_checkpoint_path).split('.')[0]
job_type = upstream_ckpt_name
if args.interpolation=='bicubic':
    job_type += '-bicubic'
job_type += f'-{args.dataset}' 
wandb_project = 'transfer-pre-extracted-no-data-aug-imgnet-stats'
wandb_project += '-' + base_arch
wandb.init(project=wandb_project,
           job_type=job_type,
           group=f'{args.dataset}',
           config=config_dictionary,
           settings=wandb.Settings(start_method="fork"))
wandb.run.name = f'seed_{args.seed}'
wandb.save(args.training_config_path)


#-----------------------------------------------------------
# Define the linear model, the trainer and progress tracker
#------------------------------------------------------------
if base_arch=='resnet50':
    linear_model = torch.nn.Linear(2048, num_classes)
elif base_arch=='mobilenet':
    linear_model = torch.nn.Linear(1024, num_classes)
elif base_arch=='resnet18':
    linear_model = torch.nn.Linear(512, num_classes)
elif base_arch=='resnet34':
    linear_model = torch.nn.Linear(512, num_classes)
else:
    raise ValueError('Choose between resnet18 | resnet34 | resnet50 | mobilenet')
linear_model.to(args.device)
wandb.watch(linear_model)
model_config = {'arch': 'linear', 'dataset': args.dataset}
trainer = build_training_policy_from_config(linear_model, args.training_config, trainer_name='default_trainer')

init_epoch = 0
print_freq = 30
training_progress = TrainingProgressTracker(init_epoch, 
                                            len(train_features_loader), 
                                            len(test_features_loader.dataset),
                                            print_freq,
                                            args.run_dir)
n_samples = len(train_features_loader.dataset)

best_acc = 0.
for epoch in range(init_epoch, args.epochs):
    epoch_trn_loss, epoch_trn_acc = 0., 0.
    start_epoch = time.time()
    for i, batch in enumerate(train_features_loader):
        start = time.time()
        loss, acc = trainer.on_minibatch_begin(batch, args.device, loss=0.)
        epoch_trn_acc += acc * batch[0].size(0) / n_samples
        epoch_trn_loss += loss.item() * batch[0].size(0) / n_samples
        trainer.on_parameter_optimization(loss, epoch, reset_momentum=False)

        training_progress.step(loss=loss,
                               acc=acc,
                               time=time.time() - start,
                               lr=trainer.optim_lr)

    wandb.log({'epoch': epoch, 'train loss': epoch_trn_loss, 'train acc': epoch_trn_acc})
    logging.info({'epoch': epoch, 'train loss': epoch_trn_loss, 'train acc': epoch_trn_acc, 'time': time.time()-start_epoch})
    trainer.on_epoch_end(bn_loader=None, swap_back=False, device=args.device, epoch_num=epoch)
    eval_loss, total_correct_eval, avg_per_class_accuracy = trainer.eval_model(test_features_loader, args.device, epoch, avg_per_class=True)
    eval_acc = total_correct_eval / len(test_features_loader.dataset)
    wandb.log({'epoch': epoch, 'val loss': eval_loss, 'val acc': eval_acc, 'val avg class acc': avg_per_class_accuracy})
    logging.info({'epoch': epoch, 'val loss': eval_loss, 'val acc': eval_acc, 'val avg class acc': avg_per_class_accuracy})
    wandb.log({'epoch': epoch, 'lr': trainer.optim_lr})
    logging.info({'epoch': epoch, 'lr': trainer.optim_lr})
    is_best_model = False
    if eval_acc > best_acc:
        best_acc = eval_acc
        is_best_model = True
    save_checkpoint(epoch, model_config, linear_model, args.run_dir,
                    is_best=is_best_model, is_scheduled_checkpoint=False)

    
