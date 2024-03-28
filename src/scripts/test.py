import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.engines.vad_engine import VadModel
from src.datasets.data_module import GlobalDataModule
from src.utils.helper import prepare_data_module_params

from config.config import *


def test_vad(**kwargs):
    """
    Test the VAD model
    
    Parameters
    ----------
     **kwargs : dict
        Keyword arguments for configuring the dataset, model and training. The following keys are supported:

        - 'model_name' (str): Name of the model to be used.
        - 'dataset_name' (str): Name of the dataset to operate on.
        - 'supported_models' (list): A list of supported model names.
        - 'supported_datasets' (list): A list of supported dataset names.
        - 'seed' (int): seed to ensure reproducibility
        - 'experiments_dir' (str): Directory path for storing experiment-related files and model checkpoints.
        - 'test_dataset_names' - (list) list of datasets to be used for testing (individually). 
            Each dataset has as separate dictionary of model-specific configuration. 
            Some of them are listed below. Others might have dataset-specific paramteres
            - 'train_cut_set_path' (str): Path to the training cut set file.
            - 'dev_cut_set_path' (str): Path to the development cut set file.
            - 'test_cut_set_path' (str): Path to the test cut set file.
        - 'max_duration' (float): Maximum duration of each batch of audio data in the dataloader in seconds.
        - 'model_dict' (dict): Dictionary of model-specific configuration.
        - 'device' (str): Device to use for processing (e.g., 'cpu' or 'gpu').
        - 'max_epochs' (int): Maximum number of training epochs for the model.
        - 'is_wandb' (bool): If True, data is logged to weights and biases, else locally
        - 'wandb' (dict): Dictionary of wandb-logging-specific configuration
            - 'project' (str): project name to log to
            - 'run' (str): current wandb run name
    """

    # torch.set_printoptions(profile="full")

    assert kwargs['model_name'] in kwargs['supported_models'], f"Invalid model {kwargs['model_name']}. Model should be one of {kwargs['supported_models']}"

    for dataset_name in kwargs['test_dataset_names']:
        assert dataset_name in kwargs['supported_datasets'], f"Invalid dataset {dataset_name}. Dataset should be one of {kwargs['supported_datasets']}"

    assert kwargs['load_checkpoint'], "Please provide a checkpoint path to load from"

    pl.seed_everything(kwargs['seed'], workers=True)

    if not os.path.exists(kwargs['experiments_dir']):
        Path(kwargs['experiments_dir']).mkdir(parents=True, exist_ok=True)

    logger = None

    model = VadModel.load_from_checkpoint(checkpoint_path=kwargs['checkpoint_path'])

    trainer = pl.Trainer(accelerator=kwargs['device'],
                        devices=1,
                        default_root_dir=kwargs['experiments_dir'],
                        logger=logger,
                        deterministic=True,
                    )

    for dataset_name in kwargs['test_dataset_names']:
        
        data_modules_params = prepare_data_module_params([dataset_name], kwargs)
        data_module = GlobalDataModule(data_modules_params, kwargs['max_duration'])
        
        trainer.test(model, data_module)

        # if dataset_name == 'dihard3':

        #     domains = ['audiobooks', 'broadcast_interview', 'clinical', 'court', 'cts', 'maptask', 'meeting', 'restaurant', 'socio_field', 'socio_lab', 'webvideo']
        #     for domain in domains:
        #         kwargs['dihard3']['test_cut_set_path'] = f'/export/c01/ashah108/vad/data/dihard3/manifests/domains/dev_cuts_feats_ssl_{domain}.jsonl.gz'
                
        #         data_modules_params = prepare_data_module_params([dataset_name], kwargs)
        #         data_module = GlobalDataModule(data_modules_params, kwargs['max_duration'])
            
        #         trainer.test(model, data_module)
        # else:
        #     data_modules_params = prepare_data_module_params([dataset_name], kwargs)
        #     data_module = GlobalDataModule(data_modules_params, kwargs['max_duration'])
        
        #     trainer.test(model, data_module)




