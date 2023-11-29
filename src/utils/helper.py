import itertools
from typing import Iterable

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

    else:
        return []