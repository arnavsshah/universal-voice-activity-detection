from src.datasets import *

from config.config import *


def prepare_data(prepare_dataset=True, get_cuts=True, **kwargs):

    assert kwargs['dataset_names'][0] in kwargs['supported_datasets'], f"Invalid dataset {kwargs['dataset_names'][0]}. Dataset should be one of {kwargs['supported_datasets']}"

    if kwargs['dataset_names'][0] == 'ami':

        data_dict = {
            'audio_dir': kwargs['ami']['audio_dir'],
            'output_dir': kwargs['ami']['output_dir'],
            'feats_dir': kwargs['ami']['feats_dir'],
            'mic': kwargs['ami']['mic'],
            'batch_duration': kwargs['ami']['batch_duration'],
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
        }

        if prepare_dataset:
            create_switchboard_dataset(**data_dict)

        if get_cuts:
            train_cuts, dev_cuts, test_cuts = create_switchboard_cut(**data_dict)

    elif kwargs['dataset_names'][0] == 'callhome_english':

        data_dict = {
            'audio_dir': kwargs['callhome_english']['audio_dir'],
            'transcript_dir': kwargs['callhome_english']['transcript_dir'],
            'output_dir': kwargs['callhome_english']['output_dir'],
            'feats_dir': kwargs['callhome_english']['feats_dir'],
            'batch_duration': kwargs['callhome_english']['batch_duration'],
        }

        if prepare_dataset:
            create_callhome_english_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_callhome_english_cut(**data_dict, phase='train')
            dev_cuts = create_callhome_english_cut(**data_dict, phase='devtest')
            test_cuts = create_callhome_english_cut(**data_dict, phase='evaltest')
    
    elif kwargs['dataset_names'][0] == 'callhome_egyptian':

        data_dict = {
            'audio_dir': kwargs['callhome_egyptian']['audio_dir'],
            'transcript_dir': kwargs['callhome_egyptian']['transcript_dir'],
            'output_dir': kwargs['callhome_egyptian']['output_dir'],
            'feats_dir': kwargs['callhome_egyptian']['feats_dir'],
            'batch_duration': kwargs['callhome_egyptian']['batch_duration'],
        }

        if prepare_dataset:
            create_callhome_egyptian_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_callhome_egyptian_cut(**data_dict, phase='train')
            dev_cuts = create_callhome_egyptian_cut(**data_dict, phase='devtest')
            test_cuts = create_callhome_egyptian_cut(**data_dict, phase='evaltest')
    
    elif kwargs['dataset_names'][0] == 'chime6':

        data_dict = {
            'corpus_dir': kwargs['chime6']['corpus_dir'],
            'output_dir': kwargs['chime6']['output_dir'],
            'mic': kwargs['chime6']['mic'],
            'feats_dir': kwargs['chime6']['feats_dir'],
            'batch_duration': kwargs['chime6']['batch_duration'],
        }

        if prepare_dataset:
            create_chime6_dataset(**data_dict)

        if get_cuts:
            # train_cuts = create_chime6_cut(**data_dict, phase='train')
            # dev_cuts = create_chime6_cut(**data_dict, phase='dev')
            test_cuts = create_chime6_cut(**data_dict, phase='eval')

    elif kwargs['dataset_names'][0] == 'fisher_english':

        data_dict = {
            'corpus_dir': kwargs['fisher_english']['corpus_dir'],
            'audio_dirs': kwargs['fisher_english']['audio_dirs'],
            'transcript_dirs': kwargs['fisher_english']['transcript_dirs'],
            'output_dir': kwargs['fisher_english']['output_dir'],
            'feats_dir': kwargs['fisher_english']['feats_dir'],
            'batch_duration': kwargs['fisher_english']['batch_duration'],
        }

        if prepare_dataset:
            create_fisher_english_dataset(**data_dict)

        # if get_cuts:
        #     train_cuts = create_fisher_english_cut(**data_dict, phase='train')
        #     dev_cuts = create_fisher_english_cut(**data_dict, phase='devtest')
        #     test_cuts = create_fisher_english_cut(**data_dict, phase='evaltest')

    elif kwargs['dataset_names'][0] == 'tedlium':

        data_dict = {
            'tedlium_root': kwargs['tedlium']['tedlium_root'],
            'output_dir': kwargs['tedlium']['output_dir'],
            'feats_dir': kwargs['tedlium']['feats_dir'],
            'batch_duration': kwargs['tedlium']['batch_duration'],
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
        }

        if prepare_dataset:
            create_voxconverse_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_voxconverse_cut(**data_dict, phase='train')
            dev_cuts = create_voxconverse_cut(**data_dict, phase='dev')
            test_cuts = create_voxconverse_cut(**data_dict, phase='test')
    
    
