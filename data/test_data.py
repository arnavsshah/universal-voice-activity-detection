import torch

from src.datasets.data_module import GlobalDataModule

from config.config import *


def test(data_modules_params, name, kwargs):
    data_module = GlobalDataModule(data_modules_params, kwargs['max_duration'])
    data_module.setup()
    dataloader = data_module.train_dataloader()

    time = 0
    # 'inputs', 'input_lens', 'is_voice', 'cut'
    for batch in dataloader:
        time += 600
    
    print(f"Total time for {name}: {time/3600} hours")


def test_data(**kwargs):

    assert kwargs['dataset_names'][0] in kwargs['supported_datasets'], f"Invalid dataset {kwargs['dataset_names'][0]}. Dataset should be one of {kwargs['supported_datasets']}"

    data_modules_params = {}

    if 'ami' in kwargs['dataset_names']:
        data_modules_params = {}
        train_data_dict = {'cut_set_path': kwargs['ami']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['ami']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['ami']['test_cut_set_path']}

        data_modules_params['ami'] = [train_data_dict, dev_data_dict, test_data_dict]
        test(data_modules_params, 'ami', kwargs)

    if 'tedlium' in kwargs['dataset_names']:
        data_modules_params = {}
        train_data_dict = {'cut_set_path': kwargs['tedlium']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['tedlium']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['tedlium']['test_cut_set_path']}

        data_modules_params['tedlium'] = [train_data_dict, dev_data_dict, test_data_dict]
        test(data_modules_params, 'tedlium', kwargs)

    if 'switchboard' in kwargs['dataset_names']:
        data_modules_params = {}
        train_data_dict = {'cut_set_path': kwargs['switchboard']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['switchboard']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['switchboard']['test_cut_set_path']}

        data_modules_params['switchboard'] = [train_data_dict, dev_data_dict, test_data_dict]
        test(data_modules_params, 'switchboard', kwargs)

    if 'voxconverse' in kwargs['dataset_names']:
        data_modules_params = {}
        train_data_dict = {'cut_set_path': kwargs['voxconverse']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['voxconverse']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['voxconverse']['test_cut_set_path']}

        data_modules_params['voxconverse'] = [train_data_dict, dev_data_dict, test_data_dict]
        test(data_modules_params, 'voxconverse', kwargs)

    if 'chime6' in kwargs['dataset_names']:
        data_modules_params = {}
        train_data_dict = {'cut_set_path': kwargs['chime6']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['chime6']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['chime6']['test_cut_set_path']}

        data_modules_params['chime6'] = [train_data_dict, dev_data_dict, test_data_dict]
        test(data_modules_params, 'chime6', kwargs)

    if 'dihard3' in kwargs['dataset_names']:
        data_modules_params = {}
        dev_data_dict = {'cut_set_path': kwargs['dihard3']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['dihard3']['test_cut_set_path']}

        data_modules_params['dihard3'] = [dev_data_dict, test_data_dict]
        test(data_modules_params, 'dihard3', kwargs)


    if 'dipco' in kwargs['dataset_names']:
        data_modules_params = {}
        train_data_dict = {'cut_set_path': kwargs['dipco']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['dipco']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['dipco']['test_cut_set_path']}

        data_modules_params['dipco'] = [train_data_dict, dev_data_dict, test_data_dict]
        test(data_modules_params, 'dipco', kwargs)

    # if 'voxceleb' in kwargs['dataset_names']:
    #     data_modules_params = {}
    #     train_data_dict = {'cut_set_path': kwargs['voxceleb']['train_cut_set_path']}
    #     dev_data_dict = {'cut_set_path': kwargs['voxceleb']['dev_cut_set_path']}
    #     test_data_dict = {'cut_set_path': kwargs['voxceleb']['test_cut_set_path']}

    #     data_modules_params['voxceleb'] = [train_data_dict, dev_data_dict, test_data_dict]
    #     test(data_modules_params, 'voxceleb', kwargs)

    # if 'mgb2' in kwargs['dataset_names']:
    #     data_modules_params = {}
    #     train_data_dict = {'cut_set_path': kwargs['mgb2']['train_cut_set_path']}
    #     dev_data_dict = {'cut_set_path': kwargs['mgb2']['dev_cut_set_path']}
    #     test_data_dict = {'cut_set_path': kwargs['mgb2']['test_cut_set_path']}

    #     data_modules_params['mgb2'] = [train_data_dict, dev_data_dict, test_data_dict]
    #     test(data_modules_params, 'mgb2', kwargs)

    if 'gale_arabic' in kwargs['dataset_names']:
        data_modules_params = {}
        train_data_dict = {'cut_set_path': kwargs['gale_arabic']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['gale_arabic']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['gale_arabic']['test_cut_set_path']}

        data_modules_params['gale_arabic'] = [train_data_dict, dev_data_dict, test_data_dict]
        test(data_modules_params, 'gale_arabic', kwargs)
    
    if 'gale_mandarin' in kwargs['dataset_names']:
        data_modules_params = {}
        train_data_dict = {'cut_set_path': kwargs['gale_mandarin']['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': kwargs['gale_mandarin']['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': kwargs['gale_mandarin']['test_cut_set_path']}

        data_modules_params['gale_mandarin'] = [train_data_dict, dev_data_dict, test_data_dict]
        test(data_modules_params, 'gale_mandarin', kwargs)


    # data_module = GlobalDataModule(data_modules_params, kwargs['max_duration'])
    # data_module.setup()
    # dataloader = data_module.train_dataloader()
        
    # batch = next(iter(dataloader))

    # torch.set_printoptions(threshold=40_000)
    # print(batch['inputs'][0])  # (120, 500/250, 80/768)
    # print(batch['input_lens'].shape)
    # print(batch['is_voice'].shape)  # (120, 500/250)
    # print(batch['cut'])
    # print(batch['is_voice'])

    # i = 1
    # for batch in dataloader:
    #     i += 1
    #     print(torch.sum(batch['inputs']))
    # print(i)

    
