from src.datasets.librispeech.utils import create_librispeech_cut

from config.config import *


def prepare_data(
    dataset: str
):

    datasets = ['librispeech']
    assert dataset in datasets, f'Invalid dataset {dataset}. Dataset should be one of {dataset}'

    if dataset == 'librispeech':
        data_dict = {
            'audio_dir': audio_dir,
            'output_dir': output_dir,
            'feats_dir': feats_dir,
            'batch_duration': batch_duration,
        }

        train_cuts = create_librispeech_cut(**data_dict, phase='train', prepare_dataset=False)
        dev_cuts = create_librispeech_cut(**data_dict, phase='dev', prepare_dataset=False)
        test_cuts = create_librispeech_cut(**data_dict, phase='test', prepare_dataset=False)



