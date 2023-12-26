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


def train_vad(**kwargs):
    """
    Train the VAD model
    
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
        - 'dataset_names' - (list) list of datasets to be used. Each dataset has as separate dictionary of model-specific configuration. 
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

    for dataset_name in kwargs['dataset_names']:
        assert dataset_name in kwargs['supported_datasets'], f"Invalid dataset {dataset_name}. Dataset should be one of {kwargs['supported_datasets']}"

    pl.seed_everything(kwargs['seed'], workers=True)

    data_modules_params = prepare_data_module_params(kwargs['dataset_names'], kwargs)
    data_module = GlobalDataModule(data_modules_params, kwargs['max_duration'], weights=kwargs['dataset_weights'], stop_early=kwargs['stop_early'])
    
    if not os.path.exists(kwargs['experiments_dir']):
        Path(kwargs['experiments_dir']).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        save_top_k=-1,
        dirpath=kwargs['experiments_dir'],
        filename="checkpoint-{epoch:02d}",
    )

    if kwargs['load_checkpoint']:
        model = VadModel.load_from_checkpoint(checkpoint_path=kwargs['checkpoint_path'])
    else:
        model = VadModel(
            kwargs['model_name'], 
            kwargs['model_dict'],
            learning_rate=kwargs['learning_rate'],
        )

    if kwargs['is_wandb']:
        logger = WandbLogger(
            project=kwargs['wandb']['project'],
            name=kwargs['wandb']['name'],
            log_model=False
        )
    else:
        logger = None

    trainer = pl.Trainer(accelerator=kwargs['device'], 
                        max_epochs=kwargs['max_epochs'], 
                        devices=1,
                        default_root_dir=kwargs['experiments_dir'],
                        logger=logger,
                        callbacks=[checkpoint_callback],
                        check_val_every_n_epoch=1,
                        deterministic=True,
                    )

    trainer.fit(model, data_module)



