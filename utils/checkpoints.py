import os
import errno
import torch
import shutil
import logging
import inspect

from models import get_model
from utils.utils import normalize_module_name, _fix_double_wrap_dict
from utils.masking_utils import (is_wrapped_layer, 
                                 WrappedLayer, 
                                 get_wrapped_model,
                                 )
import torchvision

import pdb

__all__ = ['save_eval_checkpoint', 'save_checkpoint', 
           'load_eval_checkpoint', 'load_checkpoint',
           'get_unwrapped_model']


# Add dummy stuff here if you need to add an optimizer or a lr_scheduler
# that have required constructor arguments (and you want to recover it 
# from state dict later and need some dummy value)
DUMMY_VALUE_FOR_OPT_ARG = {'lr': 1e-3, 'gamma': 0.9}


def should_unwrap_layer(layer: 'nn.Module') -> bool:
    return isinstance(layer, WrappedLayer)

def unwrap_module(module: 'nn.Module', prefix='.'):
    """
    Recursive function which iterates over WRAPPED_MODULES of this
    module and unwraps them.
    """
    module_dict = dict(module.named_children())
    for name, sub_module in module_dict.items():
        if should_unwrap_layer(sub_module):
            setattr(module, name, sub_module.unwrap())
            print(f'Module {prefix + name} was successfully unwrapped')
            continue
        unwrap_module(sub_module, prefix + name + '.')

def get_unwrapped_model(model: 'nn.Module') -> 'nn.Module':
    """
    Function which unwrappes the wrapped layers of received model.
    """
    unwrap_module(model)
    return model

def save_eval_checkpoint(model_config: str, model: 'nn.Module', checkpoint_path: str):
    """
    Save the model state dict with all layer unwrapped and 
    pruning masks applied.
    
    Arguments:
        model_config {dict} -- {'arch': arch, 'dataset': dataset}
        path {str} -- path to save wrapped model (e.g.: exps_root/sample_run/run_id)
    """
    if not os.path.isdir(checkpoint_path):
        raise IOError(errno.ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(checkpoint_path))

    model = get_unwrapped_model(model)
    if isinstance(model, torch.nn.DataParallel):
        # TODO: not sure it should be here
        logging.debug('Was using data parallel')
        model = model.module
    model_state_dict = model.state_dict()
    state_dict = dict()
    state_dict['model_config'] = model_config
    state_dict['model_state_dict'] = model_state_dict
    torch.save(state_dict, os.path.join(checkpoint_path, 'eval_ready_state_dict.ckpt'))

def load_eval_checkpoint(checkpoint_path: str) -> 'nn.Module':
    """
    Load the evaluation ready model given the chepoint path.
    """
    try:
        state_dict = torch.load(os.path.join(checkpoint_path, 'eval_ready_state_dict.ckpt'))
    except:
        raise IOError(errno.ENOENT, 'Evaluation checkpoint does not exist at', os.path.abspath(checkpoint_path))
    model_config = state_dict['model_config']
    model = get_model(model_config['arch'], model_config['dataset'])
    model.load_state_dict(state_dict['model_state_dict'])
    return model

def save_checkpoint(epoch, model_config, model,
                    checkpoint_path: str,
                    is_best=False, is_scheduled_checkpoint=False):
    """
    This function damps the full checkpoint for the running manager.
    Including the epoch, model, optimizer and lr_scheduler states. 
    """
    if not os.path.isdir(checkpoint_path):
        raise IOError(errno.ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(checkpoint_path))

    checkpoint_dict = dict()
    checkpoint_dict['epoch'] = epoch
    checkpoint_dict['model_config'] = model_config
    checkpoint_dict['model_state_dict'] = model.state_dict()

    path_regular = os.path.join(checkpoint_path, f'regular_checkpoint{epoch}.ckpt')
    path_best = os.path.join(checkpoint_path, 'best_checkpoint.ckpt')
    path_last = os.path.join(checkpoint_path, 'last_checkpoint.ckpt')
    torch.save(checkpoint_dict, path_last)
    if is_best:
        print("Saving best checkpoint.")
        shutil.copyfile(path_last, path_best)
    if is_scheduled_checkpoint:
        print("Saving checkpoint on schedule.")
        shutil.copyfile(path_last, path_regular)


def load_checkpoint(full_checkpoint_path: str, model=None, load_modules=None, dset=None, apply_deepsparse=False, wrapper_input_shape=(64, 3, 224, 224)):
    """
    Loads checkpoint give full checkpoint path.
    """
    try:
        checkpoint_dict = torch.load(full_checkpoint_path, map_location='cpu')
    except:
        raise IOError(errno.ENOENT, 'Checkpoint file does not exist at', os.path.abspath(full_checkpoint_path))
    
    model_config = checkpoint_dict['model_config']
    if dset is not None:
        model_config["dataset"] = dset
        print(model_config)
    if any(m.endswith('_layer.weight') for  m in checkpoint_dict['model_state_dict'].keys()):
        model = get_wrapped_model(get_model(**model_config, deepsparse_wrapper=apply_deepsparse, wrapper_input_shape=wrapper_input_shape))
        is_unwrapped = False
    else:
        is_unwrapped = True
        model = get_model(**model_config, deepsparse_wrapper=apply_deepsparse, wrapper_input_shape=wrapper_input_shape)

    updated_state_dict = model.state_dict()
    if 'module' in list(checkpoint_dict['model_state_dict'].keys())[0]:
        checkpoint_dict['model_state_dict'] = {normalize_module_name(k): v for k, v in checkpoint_dict['model_state_dict'].items()}
    def should_be_loaded(name):
        if load_modules is not None:
            for prefix in load_modules:
                if name.startswith(prefix):
                    return True
            return False
        else:
            return True

    if load_modules is not None:
        checkpoint_dict['model_state_dict'] = {k: v for k, v in checkpoint_dict['model_state_dict'].items() if should_be_loaded(k)}
    
    # Check that there is an exact match between model and checkpoint keys.
    model_keys = {k for k in updated_state_dict.keys()}
    checkpoint_keys = {k for k in checkpoint_dict["model_state_dict"].keys()}
    if load_modules is not None:
        in_model_not_in_checkpoint = {k for k in model_keys.difference(checkpoint_keys) if k.startswith(tuple(load_modules))}
    else:
        in_model_not_in_checkpoint = model_keys.difference(checkpoint_keys)

    in_checkpoint_not_in_model = checkpoint_keys.difference(model_keys)
    if in_model_not_in_checkpoint or in_checkpoint_not_in_model:
        raise ValueError(f"Mismatch between model and checkpoints:\n  Tensors in model not in checkpoint:{in_model_not_in_checkpoint}\n  In checkpoint:{in_checkpoint_not_in_model}") 

    for k in updated_state_dict.keys():
        if k in checkpoint_dict["model_state_dict"]:
            updated_state_dict[k] = checkpoint_dict["model_state_dict"][k]
    model.load_state_dict(updated_state_dict)

    if is_unwrapped:
        model = get_wrapped_model(model)
    
    return model_config['arch'], model
