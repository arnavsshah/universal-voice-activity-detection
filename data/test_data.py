import torch

from src.datasets.data_module import GlobalDataModule

from config.config import *


def test_data(**kwargs):

    assert kwargs['dataset_names'][0] in kwargs['supported_datasets'], f"Invalid dataset {kwargs['dataset_names'][0]}. Dataset should be one of {kwargs['supported_datasets']}"

    data_modules_params = {}

    if kwargs['dataset_names'][0] == 'ami':
        train_data_dict = {'cut_set_path': kwargs['ami']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['ami']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['ami']['test_cut_set_path']}

        data_modules_params['ami'] = [train_data_dict, dev_data_dict, test_data_dict]

    if kwargs['dataset_names'][0] == 'dihard3':
        dev_data_dict = {'cut_set_path': kwargs['dihard3']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['dihard3']['test_cut_set_path']}

        data_modules_params['dihard3'] = [dev_data_dict, test_data_dict]

    if kwargs['dataset_names'][0] == 'switchboard':
        train_data_dict = {'cut_set_path': kwargs['switchboard']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['switchboard']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['switchboard']['test_cut_set_path']}

        data_modules_params['switchboard'] = [train_data_dict, dev_data_dict, test_data_dict]

    if kwargs['dataset_names'][0] == 'callhome_english':
        train_data_dict = {'cut_set_path': kwargs['callhome_english']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['callhome_english']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['callhome_english']['test_cut_set_path']}

        data_modules_params['callhome_english'] = [train_data_dict, dev_data_dict, test_data_dict]

    if kwargs['dataset_names'][0] == 'callhome_egyptian':
        train_data_dict = {'cut_set_path': kwargs['callhome_egyptian']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['callhome_egyptian']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['callhome_egyptian']['test_cut_set_path']}

        data_modules_params['callhome_egyptian'] = [train_data_dict, dev_data_dict, test_data_dict]

    if kwargs['dataset_names'][0] == 'chime6':
        train_data_dict = {'cut_set_path': kwargs['chime6']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['chime6']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['chime6']['test_cut_set_path']}

        data_modules_params['chime6'] = [train_data_dict, dev_data_dict, test_data_dict]


    data_module = GlobalDataModule(data_modules_params, kwargs['max_duration'])
    data_module.prepare_data()
    dataloader = data_module.train_dataloader()
        
    batch = next(iter(dataloader))

    print(batch['inputs'].shape)  # (120, 500/250, 80/768)
    print(batch['input_lens'].shape)
    print(batch['is_voice'].shape)  # (120, 500/250)
    print(batch['cut'])
    print(batch['is_voice'])

    i = 1
    print(torch.sum(batch['is_voice']))
    for batch in dataloader:
        i += 1
        print(torch.sum(batch['is_voice']))
    print(i)
    
