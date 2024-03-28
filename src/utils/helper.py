import os
import itertools
import random
from pathlib import Path
from typing import Iterable, Dict, Optional, Union

from lhotse import load_manifest_lazy, CutSet, FbankConfig, Fbank, LilcomChunkyWriter

import torch
import numpy as np
from scipy.signal import medfilt


Pathlike = Union[Path, str]


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
    if dataset_name == 'dihard3':
        dev_data_dict = {'cut_set_path': config['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': config['test_cut_set_path']}

        return [dev_data_dict, test_data_dict]
    
    elif dataset_name == 'santa_barbara':
        test_data_dict = {'cut_set_path': config['test_cut_set_path']}

        return [test_data_dict]

    else:
        train_data_dict = {'cut_set_path': config['train_cut_set_path']}
        dev_data_dict = {'cut_set_path': config['dev_cut_set_path']}
        test_data_dict = {'cut_set_path': config['test_cut_set_path']}

        return [train_data_dict, dev_data_dict, test_data_dict]

    # else:
    #     return []


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


def compute_feats_for_cutset(
    cut_set_path: Pathlike,
    new_cuts_filename: Pathlike, 
    new_feats_filename: str,
    output_dir: Pathlike,
    feats_dir: Pathlike,
    batch_duration: int = 600,
) -> None:

    output_dir = Path(output_dir)
    feats_dir = Path(feats_dir)

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    cuts = CutSet.from_file(cut_set_path)

    extractor = Fbank(FbankConfig(device='cuda'))

    cuts = cuts.compute_and_store_features_batch(
        extractor=extractor,
        manifest_path=f'{output_dir}/{new_cuts_filename}',
        storage_path=f'{feats_dir}/{new_feats_filename}',
        batch_duration=batch_duration,
        num_workers=4,
        overwrite=True,
        storage_type=LilcomChunkyWriter,
    )

    # cuts.to_file(f'{output_dir}/{new_cuts_filename}')



def get_audacity_labels(recording_path: Pathlike, cuts_path: Pathlike):

    recs = load_manifest_lazy(recording_path)
    cuts = load_manifest_lazy(cuts_path)

    labels = {}

    for rec in recs:
        labels[rec.id] = []

    for cut in cuts:
        labels[cut.recording.id].append([cut.start, cut.start + cut.duration])
    

    for rec_id in labels:
        with open(f'/export/c01/ashah108/vad/data/audacity/{rec_id}.txt', 'w') as f:
            for start, end in labels[rec_id]:
                f.write(f'{round(start, 2)}\t{round(end, 2)}\tSPC\n')




def get_cuts_subset(cut_set_path: Pathlike, cut_subset_size: int) -> CutSet:
    cuts = CutSet.from_file(cut_set_path)
    cuts_subset_arr = []
    NUM_CUTS_PER_HOUR = 60 * 60 // 5  # 720 (as duration of each cut is 5 secs)

    original_cuts_len = len(cuts)
    original_cuts_total_time = original_cuts_len * 5

    subset_cuts_total_time = cut_subset_size * 60 * 60

    valid_cuts_total_time = min(original_cuts_total_time, subset_cuts_total_time)
    
    if valid_cuts_total_time < subset_cuts_total_time:
        print(f'Warning: Subset size is larger than the original cut set. Using the original cut set size instead.')
        cuts.to_file(str(cut_set_path).replace('.jsonl.gz', f'_subset_{cut_subset_size}_balanced.jsonl.gz'))
        return

    valid_cuts_len = int(valid_cuts_total_time / 5)

    step = original_cuts_len // cut_subset_size
    
    cut_indices = []
    for i in range(0, original_cuts_len, step):
        cut_indices += list(range(i, i + NUM_CUTS_PER_HOUR))

    for i, cut in enumerate(cuts):
        if i in cut_indices:
            cuts_subset_arr.append(cut)
    
    cuts_subset = CutSet.from_cuts(cuts_subset_arr)
    
    new_cut_set_path = str(cut_set_path).replace('.jsonl.gz', f'_subset_{cut_subset_size}_balanced.jsonl.gz')
    cuts_subset.to_file(new_cut_set_path)
    




# if __name__ == '__main__':
    # get_audacity_labels('/export/c01/ashah108/vad/data/custom/manifests/sbcsae_rec.jsonl.gz', '/export/c01/ashah108/vad/data/custom/manifests/sbcsae_cuts_collar-0.jsonl.gz')