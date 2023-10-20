import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.engines.vad_engine import VadModel
from src.datasets.librispeech.dataset import LibrispeechDataModule

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
        - 'experiments_dir' (str): Directory path for storing experiment-related files and model checkpoints.
        - [dataset_name] - each dataset has as separate dictionary of model-specific configuration.
            - 'audio_dir' (str): Directory path where audio data is located.
            - 'output_dir' (str): Directory path for saving the output of the dataset preparation (manifests).
            - 'feats_dir' (str): Directory path for storing audio features.
            - 'batch_duration' (int): Duration of each batch while preparing batched data in seconds.
            - 'train_cut_set_path' (str): Path to the training cut set file.
            - 'dev_cut_set_path' (str): Path to the development cut set file.
            - 'test_cut_set_path' (str): Path to the test cut set file.
            - 'max_duration' (float): Maximum duration for of each batch of audio data in the dataloader in seconds.
        - 'model_dict' (dict): Dictionary of model-specific configuration.
        - 'device' (str): Device to use for processing (e.g., 'cpu' or 'gpu').
        - 'max_epochs' (int): Maximum number of training epochs for the model.
        - 'is_wandb' (bool): If True, data is logged to weights and biases, else locally
        - 'wandb' (dict): Dictionary of wandb-logging-specific configuration
            - 'project' (str): project name to log to
            - 'run' (str): current wandb run name
    """


    assert kwargs['model_name'] in kwargs['supported_models'], f"Invalid model {kwargs['model_name']}. Model should be one of {kwargs['supported_models']}"
    assert kwargs['dataset_name'] in kwargs['supported_datasets'], f"Invalid dataset {kwargs['dataset_name']}. Dataset should be one of {kwargs['supported_datasets']}"

    if not os.path.exists(kwargs['experiments_dir']):
        kwargs['experiments_dir'].mkdir(parents=True, exist_ok=True)
        
    if kwargs['dataset_name'] == 'librispeech':
        data_dict = {
            'audio_dir': kwargs['librispeech']['audio_dir'],
            'output_dir': kwargs['librispeech']['output_dir'],
            'feats_dir': kwargs['librispeech']['feats_dir'],
            'batch_duration': kwargs['librispeech']['batch_duration'],
        }
        train_data_dict = {'cut_set_path': kwargs['librispeech']['train_cut_set_path'], **data_dict}
        dev_data_dict = {'cut_set_path': kwargs['librispeech']['dev_cut_set_path'], **data_dict}
        test_data_dict = {'cut_set_path': kwargs['librispeech']['test_cut_set_path'], **data_dict}

        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=1,
            dirpath=kwargs['experiments_dir'], 
            filename="checkpoint-{epoch:02d}",
        )

        librispeech = LibrispeechDataModule(train_data_dict, dev_data_dict, test_data_dict, kwargs['librispeech']['max_duration'])
        
        if kwargs['from_checkpoint']:
            model = VadModel.load_from_checkpoint(checkpoint_path=kwargs['checkpoint_path'])
        else:
            model = VadModel(kwargs['model_name'], kwargs['model_dict'])

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
                            callbacks=[checkpoint_callback])
        trainer.fit(model, librispeech)



