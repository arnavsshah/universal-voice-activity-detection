from src.datasets import *

from config.config import *


def download_data(**kwargs):

    assert kwargs['dataset_names'][0] in kwargs['supported_datasets'], f"Invalid dataset {kwargs['dataset_names'][0]}. Dataset should be one of {kwargs['supported_datasets']}"

    if kwargs['dataset_names'][0] == 'dipco':

        data_dict = {
            'target_dir': kwargs['dipco']['target_dir'],
        }
        
        download_dipco_dataset(**data_dict)
        