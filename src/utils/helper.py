import itertools
from typing import Iterable

import torch
import numpy as np
from scipy.signal import medfilt

def pairwise(iterable: Iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def merge_dict(defaults: dict, custom: dict = None):
    """merge 2 dictionaries, with the custom dict values overriding the default dict values"""
    params = dict(defaults)
    if custom is not None:
        params.update(custom)
    return params

def prepare_data_module_params(dataset_names, config):
    data_modules_params = {}

    for dataset_name in dataset_names:
        data_modules_params[dataset_name] = prepare_data_dicts(dataset_name, config[dataset_name])
    
    return data_modules_params

def prepare_data_dicts(dataset_name, config):
    if dataset_name == 'ami':
        train_data_dict = {'cut_set_path': config['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': config['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': config['test_cut_set_path']}

        return [train_data_dict, dev_data_dict, test_data_dict]

    elif dataset_name == 'dihard3':
        dev_data_dict = {'cut_set_path': config['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': config['test_cut_set_path']}

        return [dev_data_dict, test_data_dict]

    elif dataset_name == 'switchboard':
        train_data_dict = {'cut_set_path': config['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': config['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': config['test_cut_set_path']}

        return [train_data_dict, dev_data_dict, test_data_dict]
    
    elif dataset_name == 'callhome_english':
        train_data_dict = {'cut_set_path': config['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': config['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': config['test_cut_set_path']}

        return [train_data_dict, dev_data_dict, test_data_dict]

    elif dataset_name == 'callhome_egyptian':
        train_data_dict = {'cut_set_path': config['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': config['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': config['test_cut_set_path']}

        return [train_data_dict, dev_data_dict, test_data_dict]

    elif dataset_name == 'chime6':
        train_data_dict = {'cut_set_path': config['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': config['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': config['test_cut_set_path']}

        return [train_data_dict, dev_data_dict, test_data_dict]

    elif dataset_name == 'tedlium':
        train_data_dict = {'cut_set_path': config['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': config['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': config['test_cut_set_path']}

        return [train_data_dict, dev_data_dict, test_data_dict]

    elif dataset_name == 'voxconverse':
        train_data_dict = {'cut_set_path': config['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': config['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': config['test_cut_set_path']}

        return [train_data_dict, dev_data_dict, test_data_dict]

    else:
        return []


def median_filter(x, SPEECH_WINDOW=0.5, window=0.02):
    '''
    Apply median filter to input after converting it to a binary tensor based on a threshold of 0.5

    Parameters
    ----------
    y : torch.Tensor
        Input tensor
    SPEECH_WINDOW : float, optional
        Speech window, by default 0.5
    window : float, optional
        Window size, by default 0.02

    Returns
    -------
    torch.Tensor
        Median filtered output
    '''

    median_window = int(SPEECH_WINDOW / window)
    if median_window % 2 == 0 : 
        median_window = median_window - 1

    x = torch.where(x < 0.5, 0, 1).cpu()

    for i in range(len(x)):
        temp = medfilt(x[i], kernel_size=median_window)
        x[i] = torch.from_numpy(temp)
    
    x = x.to('cuda')
    
    return x