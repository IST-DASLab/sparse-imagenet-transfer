"""
Managing class for different Policies.

"""

from models import get_model
from policies.trainers import build_training_policy_from_config

from utils import read_config, get_datasets, get_wrapped_model, get_total_sparsity
from utils import (preprocess_for_device,
                   load_checkpoint,
                   save_checkpoint,
                   classification_num_classes,
                   TrainingProgressTracker)
from utils.masking_utils import WrappedLayer

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import logging

import pdb

USE_TQDM = True
if not USE_TQDM:
    tqdm = lambda x: x

DEFAULT_TRAINER_NAME = "default_trainer"




# noinspection SyntaxError
class Manager:
    """
    Class for managing mode training.
    """

    def __init__(self, args):
        args = preprocess_for_device(args)


        # REFACTOR: Model setup
        if args.manual_seed:
            torch.manual_seed(args.manual_seed)
            np.random.seed(args.manual_seed)
        else:
            np.random.seed(0)

        self.transfer_config = None
        if args.transfer_config_path:
            self.transfer_config = args.transfer_config_path if isinstance(args.transfer_config_path, dict) else read_config(args.transfer_config_path)
        
        if args.from_checkpoint_path is not None:
            num_classes = classification_num_classes(args.dset)
            arch, self.model = load_checkpoint(args.from_checkpoint_path, load_modules=self.transfer_config["load_modules"] if self.transfer_config else None,
                                                dset=args.dset, apply_deepsparse=hasattr(args, "apply_deepsparse") and args.apply_deepsparse, 
                                                wrapper_input_shape=(args.batch_size, 3, 224, 224))
            self.model_config = {'arch': arch, 'dataset': args.dset}
            args.arch = arch
        else:
            self.model_config = {'arch': args.arch, 'dataset': args.dset}
            self.model = get_model(args.arch, dataset=args.dset, pretrained=args.pretrained,
                                   deepsparse_wrapper=hasattr(args, "apply_deepsparse") and args.apply_deepsparse,
                                   wrapper_input_shape=(args.batch_size, 3, 224, 224))

            self.model = get_wrapped_model(self.model)
       
        
        # fix for RigL models with deepsparse (no need for separate config files)
        if args.apply_deepsparse and args.arch=='resnet50_rigl':
            args.interpolation = 'bicubic'
        # fix for STR models with deepsparse  
        if args.apply_deepsparse and args.arch=='resnet50_str': 
            n_classes = classification_num_classes(args.dset)
            self.model._post_module[0] = torch.nn.Linear(2048, n_classes)


        self.config = args.config_path if isinstance(args.config_path, dict) else read_config(args.config_path)

        self.logging_function = print
        if args.use_wandb:
            import wandb
            self.logging_function = wandb.log
            wandb.watch(self.model)

        self.data = (args.dset, args.dset_path)
        self.n_epochs = args.epochs
        self.num_workers = args.workers
        self.batch_size = args.batch_size
        self.steps_per_epoch = args.steps_per_epoch
        self.device = args.device
        self.initial_epoch = 0
        self.training_stats_freq = args.training_stats_freq
        self.best_val_acc = 0

         # for mixed precision training:
        if args.fp16:
            self.fp16_scaler = torch.cuda.amp.GradScaler()
            logging.info("==============!!!!Train with mixed precision (FP 16 enabled)!!!!================")
        else:
            self.fp16_scaler = None


        # Define datasets
        self.data_train, self.data_test = get_datasets(*self.data, use_data_aug=True, interpolation=args.interpolation)

        self.train_loader = DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        self.test_loader = DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)
        optimizer_dict, lr_scheduler_dict = None, None
        
        # Freeze any modules that we don't plan to train.
        update_modules = None
        if self.transfer_config is not None:
            update_modules = set()
            frozen_modules = set()
            prefix = ''
            if args.apply_deepsparse:
                prefix += '_orig_module.'
            for name, p in self.model.named_parameters():
                frozen = False
                for m in self.transfer_config["freeze_modules"]:
                    if p.requires_grad and name.startswith(prefix + m):
                        p.requires_grad = False
                        frozen = True
                        break
                if not frozen:
                    update_modules.add(name)
                else:
                    frozen_modules.add(name)
            update_modules = list(update_modules)


        if args.device.type == 'cuda':
            torch.cuda.manual_seed(args.manual_seed)
            self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus)
            self.model.to(args.device)
        

        self.track_weight_hist = args.track_weight_hist

        self.trainer = build_training_policy_from_config(self.model, self.config, 
                                                         trainer_name=DEFAULT_TRAINER_NAME,
                                                         fp16_scaler=self.fp16_scaler,
                                                         update_modules=update_modules)
        if optimizer_dict is not None and lr_scheduler_dict is not None:
            self.initial_epoch = epoch
            self.trainer.optimizer.load_state_dict(optimizer_dict)
            self.trainer.lr_scheduler.load_state_dict(lr_scheduler_dict)
        
        self.setup_logging(args)



    def setup_logging(self, args):
        self.logging_level = args.logging_level
        self.checkpoint_freq = args.checkpoint_freq
        self.exp_dir = args.exp_dir
        self.run_dir = args.run_dir

   
    def measure_sparsity(self):
       sparsity_dicts = dict()
       for name, module in self.model.named_modules():
           if isinstance(module, WrappedLayer):
               num_zeros, num_params = get_total_sparsity(module)
               sparsity_dicts[name] = (num_zeros, num_params)
       return sparsity_dicts

    def apply_masks(self):
        for module in self.model.modules():
            if isinstance(module, WrappedLayer):
                module.weight.data *= module.weight_mask


    def end_epoch(self, epoch):
        sparsity_dicts = self.measure_sparsity()
        num_zeros, num_params = get_total_sparsity(self.model)

        self.training_progress.sparsity_info(epoch,
                                             [sparsity_dicts],
                                             num_zeros,
                                             num_params,
                                             self.logging_function)
        self.trainer.on_epoch_end(device=self.device, epoch_num=epoch)


    def get_eval_stats(self, epoch, dataloader, type = 'val', avg_per_class=False):
        loss, correct, avg_per_class_acc = self.trainer.eval_model(loader=dataloader, device=self.device, 
                                                                   epoch_num=epoch, avg_per_class=avg_per_class)
        # log validation stats
        if type == 'val':
            self.logging_function({'epoch': epoch, type + ' loss':loss, type + ' acc':correct / len(dataloader.dataset), type + " avg class acc":avg_per_class_acc})
        logging.info({'epoch': epoch, type + ' loss':loss, type + ' acc':correct / len(dataloader.dataset), type+" avg class acc": avg_per_class_acc})

        if type == "val":
            self.training_progress.val_info(epoch, loss, correct)
        return loss, correct


    def run(self):
        self.training_progress = TrainingProgressTracker(self.initial_epoch,
                                                         len(self.train_loader),
                                                         len(self.test_loader.dataset),
                                                         self.training_stats_freq,
                                                         self.run_dir)


        for epoch in range(self.initial_epoch, self.n_epochs):
            logging.info(f"Starting epoch {epoch} with number of batches {self.steps_per_epoch or len(self.train_loader)}")

            # the following line is just to ensure reproducibility compared to old runs
            # will be removed in the final version
            subset_inds = np.random.choice(len(self.data_train), 1000, replace=False)

            epoch_loss, epoch_acc = 0., 0.
            n_samples = len(self.train_loader.dataset)
            for i, batch in enumerate(self.train_loader):
                if self.steps_per_epoch and i > self.steps_per_epoch:
                    break
                start = time.time()
                
                loss, acc = self.trainer.on_minibatch_begin(minibatch=batch, device=self.device, loss=0.)
                
                epoch_acc += acc * batch[0].size(0) / n_samples
                epoch_loss += loss.item() * batch[0].size(0) / n_samples

                self.trainer.on_parameter_optimization(loss=loss, epoch_num=epoch)
                
                self.apply_masks()

                #tracking the training statistics
                self.training_progress.step(loss=loss, acc=acc, time=time.time() - start, lr=self.trainer.optim_lr)
                #############################################################################################

            # log train stats
            self.logging_function({'epoch': epoch, 'train loss': epoch_loss, 'train acc': epoch_acc})

            self.end_epoch(epoch)

            ####################################################

            current_lr = self.trainer.optim_lr
            self.logging_function({'epoch': epoch, 'lr': current_lr})
            
            val_loss, val_correct = self.get_eval_stats(epoch, self.test_loader, avg_per_class=True)

            val_acc = 1.0 * val_correct / len(self.test_loader.dataset)
            is_best = False
            scheduled = False
            if val_acc > self.best_val_acc:
                is_best = True
                self.best_val_acc = val_acc
            if (epoch + 1)  % self.checkpoint_freq == 0:
                logging.info("scheduled checkpoint")
                scheduled = True
            save_checkpoint(epoch, self.model_config, self.model, self.run_dir,
                            is_best = is_best,
                            is_scheduled_checkpoint=scheduled)

        # end_epoch even if epochs==0
        logging.info('====>Final summary for the run:')
        self.end_epoch(self.n_epochs)



        return val_correct, len(self.test_loader.dataset)


    def run_eval(self):

        loss, correct, avg_per_class_acc = self.trainer.eval_model(
                loader=self.test_loader, device=self.device, 
                epoch_num=0, avg_per_class=True)
        # log validation stats
        # TODO: also add sparsity info.
        # overall_sparsity =np.sum([x[0].cpu().numpy() for x in layer_sparsities.values()])/np.sum([x[1] for x in layer_sparsities.values()])
        logging.info({'val loss':loss,  'val acc':correct / len(self.test_loader.dataset), "val avg class acc": avg_per_class_acc})#, "overall sparsity": overall_sparsity})


