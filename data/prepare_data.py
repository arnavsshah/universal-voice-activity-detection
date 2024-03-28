from src.datasets import *
from src.utils.helper import compute_feats_for_cutset, get_cuts_subset 
from config.config import *


def prepare_feats(**kwargs):
    assert kwargs['dataset_names'][0] in kwargs['supported_datasets'], f"Invalid dataset {kwargs['dataset_names'][0]}. Dataset should be one of {kwargs['supported_datasets']}"

    data_dict = {
        'cut_set_path': kwargs['compute_feats']['cut_set_path'],
        'new_cuts_filename': kwargs['compute_feats']['new_cuts_filename'],
        'new_feats_filename': kwargs['compute_feats']['new_feats_filename'],
        'output_dir': kwargs['compute_feats']['output_dir'],
        'feats_dir': kwargs['compute_feats']['feats_dir'],
        'batch_duration': kwargs['compute_feats']['batch_duration'],
    }

    compute_feats_for_cutset(**data_dict)


def prepare_cuts_subset(**kwargs) -> None:

    for dataset_name in kwargs['dataset_names']:

        assert dataset_name in kwargs['supported_datasets'], f"Invalid dataset {dataset_name}. Dataset should be one of {kwargs['supported_datasets']}"

        if dataset_name == 'dihard3':
            get_cuts_subset(kwargs['dihard3']['dev_cut_set_path'], kwargs['cut_subset_size'])
        else:
            get_cuts_subset(kwargs[dataset_name]['train_cut_set_path'], kwargs['cut_subset_size'])



def prepare_data(prepare_dataset=True, get_cuts=True, **kwargs):

    assert kwargs['dataset_names'][0] in kwargs['supported_datasets'], f"Invalid dataset {kwargs['dataset_names'][0]}. Dataset should be one of {kwargs['supported_datasets']}"

    if kwargs['dataset_names'][0] == 'musan':
        data_dict = {
            'corpus_dir': kwargs['musan']['corpus_dir'],
            'output_dir': kwargs['musan']['output_dir'],
            'feats_dir': kwargs['musan']['feats_dir'],
            'batch_duration': kwargs['musan']['batch_duration'],
            'feature_extractor': kwargs['feature_extractor'],
        }

        if prepare_dataset:
            create_musan_dataset(**data_dict)
        
        if get_cuts:
            cuts = create_musan_cut(**data_dict)
        
    elif kwargs['dataset_names'][0] == 'ami':

        data_dict = {
            'audio_dir': kwargs['ami']['audio_dir'],
            'output_dir': kwargs['ami']['output_dir'],
            'feats_dir': kwargs['ami']['feats_dir'],
            'mic': kwargs['ami']['mic'],
            'batch_duration': kwargs['ami']['batch_duration'],
            'feature_extractor': kwargs['feature_extractor'],
        }

        if prepare_dataset:
            create_ami_dataset(**data_dict)
        
        if get_cuts:
            train_cuts = create_ami_cut(**data_dict, phase='train')
            dev_cuts = create_ami_cut(**data_dict, phase='dev')
            test_cuts = create_ami_cut(**data_dict, phase='test')
        
    elif kwargs['dataset_names'][0] == 'dihard3':

        data_dict = {
            'dev_audio_dir': kwargs['dihard3']['dev_audio_dir'],
            'eval_audio_dir': kwargs['dihard3']['eval_audio_dir'],
            'output_dir': kwargs['dihard3']['output_dir'],
            'feats_dir': kwargs['dihard3']['feats_dir'],
            'batch_duration': kwargs['dihard3']['batch_duration'],
            'feature_extractor': kwargs['feature_extractor'],
        }

        if prepare_dataset:
            create_dihard3_dataset(**data_dict)

        if get_cuts:
            # train_cuts = create_dihard3_cut(**data_dict, phase='train')
            dev_cuts = create_dihard3_cut(**data_dict, phase='dev')
            test_cuts = create_dihard3_cut(**data_dict, phase='eval')
    
    elif kwargs['dataset_names'][0] == 'switchboard':

        data_dict = {
            'audio_dir': kwargs['switchboard']['audio_dir'],
            'transcripts_dir': kwargs['switchboard']['transcripts_dir'],
            'output_dir': kwargs['switchboard']['output_dir'],
            'feats_dir': kwargs['switchboard']['feats_dir'],
            'batch_duration': kwargs['switchboard']['batch_duration'],
            'feature_extractor': kwargs['feature_extractor'],
        }

        if prepare_dataset:
            create_switchboard_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_switchboard_cut(**data_dict, phase='train')
            dev_cuts = create_switchboard_cut(**data_dict, phase='dev')

    
    elif kwargs['dataset_names'][0] == 'eval2000':

        data_dict = {
            'corpus_dir': kwargs['eval2000']['corpus_dir'],
            'output_dir': kwargs['eval2000']['output_dir'],
            'feats_dir': kwargs['eval2000']['feats_dir'],
            'batch_duration': kwargs['eval2000']['batch_duration'],
            'feature_extractor': kwargs['feature_extractor'],
        }

        if prepare_dataset:
            create_eval2000_dataset(**data_dict)

        if get_cuts:
            test_cuts = create_eval2000_cut(**data_dict)

    
    elif kwargs['dataset_names'][0] == 'chime6':

        data_dict = {
            'corpus_dir': kwargs['chime6']['corpus_dir'],
            'output_dir': kwargs['chime6']['output_dir'],
            'mic': kwargs['chime6']['mic'],
            'feats_dir': kwargs['chime6']['feats_dir'],
            'batch_duration': kwargs['chime6']['batch_duration'],
            'feature_extractor': kwargs['feature_extractor'],
        }

        if prepare_dataset:
            create_chime6_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_chime6_cut(**data_dict, phase='train')
            dev_cuts = create_chime6_cut(**data_dict, phase='dev')
            test_cuts = create_chime6_cut(**data_dict, phase='eval')


    elif kwargs['dataset_names'][0] == 'tedlium':

        data_dict = {
            'tedlium_root': kwargs['tedlium']['tedlium_root'],
            'output_dir': kwargs['tedlium']['output_dir'],
            'feats_dir': kwargs['tedlium']['feats_dir'],
            'batch_duration': kwargs['tedlium']['batch_duration'],
            'feature_extractor': kwargs['feature_extractor'],
        }

        if prepare_dataset:
            create_tedlium_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_tedlium_cut(**data_dict, phase='train')
            dev_cuts = create_tedlium_cut(**data_dict, phase='dev')
            test_cuts = create_tedlium_cut(**data_dict, phase='test')

    elif kwargs['dataset_names'][0] == 'voxconverse':

        data_dict = {
            'corpus_dir': kwargs['voxconverse']['corpus_dir'],
            'output_dir': kwargs['voxconverse']['output_dir'],
            'feats_dir': kwargs['voxconverse']['feats_dir'],
            'batch_duration': kwargs['voxconverse']['batch_duration'],
            'feature_extractor': kwargs['feature_extractor'],
        }

        if prepare_dataset:
            create_voxconverse_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_voxconverse_cut(**data_dict, phase='train')
            dev_cuts = create_voxconverse_cut(**data_dict, phase='dev')
            test_cuts = create_voxconverse_cut(**data_dict, phase='test')

    
    elif kwargs['dataset_names'][0] == 'dipco':

        data_dict = {
            'corpus_dir': kwargs['dipco']['corpus_dir'],
            'output_dir': kwargs['dipco']['output_dir'],
            'mic': kwargs['dipco']['mic'],
            'feats_dir': kwargs['dipco']['feats_dir'],
            'batch_duration': kwargs['dipco']['batch_duration'],
            'feature_extractor': kwargs['feature_extractor'],
        }

        if prepare_dataset:
            create_dipco_dataset(**data_dict)

        if get_cuts:
            train_cuts, dev_cuts = create_dipco_cut(**data_dict, phase='dev')
            test_cuts, _ = create_dipco_cut(**data_dict, phase='eval')

    
    elif kwargs['dataset_names'][0] == 'gale_arabic':

        data_dict = {
            'audio_dirs': kwargs['gale_arabic']['audio_dirs'],
            'transcript_dirs': kwargs['gale_arabic']['transcript_dirs'],
            'output_dir': kwargs['gale_arabic']['output_dir'],
            'feats_dir': kwargs['gale_arabic']['feats_dir'],
            'batch_duration': kwargs['gale_arabic']['batch_duration'],
            'feature_extractor': kwargs['feature_extractor'],
        }

        if prepare_dataset:
            create_gale_arabic_dataset(**data_dict)

        if get_cuts:
            train_cuts, dev_cuts = create_gale_arabic_cut(**data_dict, phase='train')
            test_cuts, _ = create_gale_arabic_cut(**data_dict, phase='test')


    elif kwargs['dataset_names'][0] == 'gale_mandarin':

        data_dict = {
            'audio_dirs': kwargs['gale_mandarin']['audio_dirs'],
            'transcript_dirs': kwargs['gale_mandarin']['transcript_dirs'],
            'output_dir': kwargs['gale_mandarin']['output_dir'],
            'feats_dir': kwargs['gale_mandarin']['feats_dir'],
            'batch_duration': kwargs['gale_mandarin']['batch_duration'],
            'feature_extractor': kwargs['feature_extractor'],
        }

        if prepare_dataset:
            create_gale_mandarin_dataset(**data_dict)

        if get_cuts:
            train_cuts, dev_cuts = create_gale_mandarin_cut(**data_dict, phase='train')
            test_cuts, _ = create_gale_mandarin_cut(**data_dict, phase='dev')


    elif kwargs['dataset_names'][0] == 'santa_barbara':

        data_dict = {
            'output_dir': kwargs['santa_barbara']['output_dir'],
            'feats_dir': kwargs['santa_barbara']['feats_dir'],
            'batch_duration': kwargs['santa_barbara']['batch_duration'],
            'feature_extractor': kwargs['feature_extractor'],
        }

        if prepare_dataset:
            create_santa_barbara_dataset(**data_dict)

        if get_cuts:
            test_cuts = create_santa_barbara_cut(**data_dict)
    
    
