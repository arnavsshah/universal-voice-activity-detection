import os
from pathlib import Path
from copy import deepcopy
import json

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.engines.vad_engine import VadModel
from src.datasets.data_module import GlobalDataModule
from src.utils.helper import prepare_data_module_params, median_filter

from config.config import *


def predict_vad(**kwargs):
    """
    Predict with the VAD model
    
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
        - 'predict_dataset_names' - (list) list of datasets to be used for prediction (for now, one only). 
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

    for dataset_name in kwargs['predict_dataset_names']:
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

    data_modules_params = prepare_data_module_params(kwargs['predict_dataset_names'], kwargs)
    data_module = GlobalDataModule(data_modules_params, kwargs['max_duration'])
    data_module.prepare_data()

    preds = trainer.predict(model, data_module.test_dataloader())  # list of (batch_size, num_frames, 1)

    # preds = median_filter(preds.squeeze(-1))  # (batch, num_frames)
    # preds = preds.unsqueeze(-1)  # (batch, num_frames, 1)

    print(len(preds))

    pred_cuts = list(data_module.test_cuts.to_dicts())

    frame_shift = 0.02  # if ssl, else 0.01
    # num_frames = 250  # if ssl, else 500

    # this adds disjoint supervisions - no overlap between 2 cuts (so no negative start time)
    for i, pred in enumerate(preds):
        # (batch_size, num_frames, 1)

        for j, p in enumerate(pred):
            # (num_frames, 1)

            cut_index = i * kwargs['batch_size'] + j
            
            if 'supervisions' in pred_cuts[cut_index]:
                pred_cuts[cut_index]['supervisions'] = []
            
                recording_id = pred_cuts[cut_index]['recording']['id']
                channel = 0

                intervals = []
                start = None
                for k, value in enumerate(p):
                    if value >= 0.5:  # threshold 0.5
                        if start is None:
                            start = k * frame_shift
                    else:
                        if start is not None:
                            end = (k-1) * frame_shift
                            intervals.append((start, end))
                            start = None
                if start is not None:
                    end = (len(p)-1) * frame_shift
                    intervals.append((start, end))

                supervision_index = 0
                for interval in intervals:
                    pred_cuts[cut_index]['supervisions'].append({
                        'id': f'{recording_id}_{channel}_{supervision_index}',
                        'recording_id': recording_id,
                        'start': interval[0],
                        'duration': interval[1] - interval[0],
                        'channel': channel,
                    })

                    supervision_index += 1
                
            else:
                for j, obj in enumerate(pred_cuts[cut_index]['tracks']):
                    
                    if 'supervisions' in obj['cut']:
                        obj['cut']['supervisions'] = []
                        recording_id = obj['cut']['recording']['id']
                        channel = 0

                        intervals = []
                        start = None
                        for k, value in enumerate(p):
                            if value >= 0.5:  # threshold 0.5
                                if start is None:
                                    start = k * frame_shift
                            else:
                                if start is not None:
                                    end = (k-1) * frame_shift
                                    intervals.append((start, end))
                                    start = None
                        if start is not None:
                            end = (len(p)-1) * frame_shift
                            intervals.append((start, end))

                        supervision_index = 0
                        for interval in intervals:
                            obj['cut']['supervisions'].append({
                                'id': f'{recording_id}_{channel}_{supervision_index}',
                                'recording_id': recording_id,
                                'start': interval[0],
                                'duration': interval[1] - interval[0],
                                'channel': channel,
                            })

                            supervision_index += 1
                        
                        pred_cuts[cut_index]['tracks'][j] = obj

    
    if not os.path.exists(kwargs['predict_output_dir']):
        Path(kwargs['predict_output_dir']).mkdir(parents=True, exist_ok=True)

    # store pred_cuts in a jsonl file
    with open(os.path.join(kwargs['predict_output_dir'], 'pred.jsonl'), 'w') as f:
        for pred_cut in pred_cuts:
            f.write(json.dumps(pred_cut) + '\n')





