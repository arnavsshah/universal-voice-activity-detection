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

        train_cuts = create_ami_cut(**data_dict, phase='train', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        dev_cuts = create_ami_cut(**data_dict, phase='dev', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        test_cuts = create_ami_cut(**data_dict, phase='test', prepare_dataset=prepare_dataset, get_cuts=get_cuts)


    elif kwargs['dataset_names'][0] == 'dihard3':

        data_dict = {
            'dev_audio_dir': kwargs['dihard3']['dev_audio_dir'],
            'eval_audio_dir': kwargs['dihard3']['eval_audio_dir'],
            'output_dir': kwargs['dihard3']['output_dir'],
            'feats_dir': kwargs['dihard3']['feats_dir'],
            'batch_duration': kwargs['dihard3']['batch_duration'],
        }

        # train_cuts = create_dihard3_cut(**data_dict, phase='train', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        dev_cuts = create_dihard3_cut(**data_dict, phase='dev', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        test_cuts = create_dihard3_cut(**data_dict, phase='eval', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
    
    elif kwargs['dataset_names'][0] == 'switchboard':

        data_dict = {
            'audio_dir': kwargs['switchboard']['audio_dir'],
            'transcripts_dir': kwargs['switchboard']['transcripts_dir'],
            'output_dir': kwargs['switchboard']['output_dir'],
            'feats_dir': kwargs['switchboard']['feats_dir'],
            'batch_duration': kwargs['switchboard']['batch_duration'],
        }

        train_cuts, dev_cuts, test_cuts = create_switchboard_cut(**data_dict, prepare_dataset=prepare_dataset, get_cuts=get_cuts)

    elif kwargs['dataset_names'][0] == 'callhome_english':

        data_dict = {
            'audio_dir': kwargs['callhome_english']['audio_dir'],
            'transcript_dir': kwargs['callhome_english']['transcript_dir'],
            'output_dir': kwargs['callhome_english']['output_dir'],
            'feats_dir': kwargs['callhome_english']['feats_dir'],
            'batch_duration': kwargs['callhome_english']['batch_duration'],
        }

        train_cuts = create_callhome_english_cut(**data_dict, phase='train', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        dev_cuts = create_callhome_english_cut(**data_dict, phase='devtest', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        test_cuts = create_callhome_english_cut(**data_dict, phase='evaltest', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
    
    elif kwargs['dataset_names'][0] == 'callhome_egyptian':

        data_dict = {
            'audio_dir': kwargs['callhome_egyptian']['audio_dir'],
            'transcript_dir': kwargs['callhome_egyptian']['transcript_dir'],
            'output_dir': kwargs['callhome_egyptian']['output_dir'],
            'feats_dir': kwargs['callhome_egyptian']['feats_dir'],
            'batch_duration': kwargs['callhome_egyptian']['batch_duration'],
        }

        train_cuts = create_callhome_egyptian_cut(**data_dict, phase='train', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        dev_cuts = create_callhome_egyptian_cut(**data_dict, phase='devtest', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        test_cuts = create_callhome_egyptian_cut(**data_dict, phase='evaltest', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
    
    elif kwargs['dataset_names'][0] == 'chime6':

        data_dict = {
            'corpus_dir': kwargs['chime6']['corpus_dir'],
            'output_dir': kwargs['chime6']['output_dir'],
            'mic': kwargs['chime6']['mic'],
            'feats_dir': kwargs['chime6']['feats_dir'],
            'batch_duration': kwargs['chime6']['batch_duration'],
        }

        train_cuts = create_chime6_cut(**data_dict, phase='train', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        dev_cuts = create_chime6_cut(**data_dict, phase='dev', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        test_cuts = create_chime6_cut(**data_dict, phase='eval', prepare_dataset=prepare_dataset, get_cuts=get_cuts)

    elif kwargs['dataset_names'][0] == 'gigaspeech':

        data_dict = {
            'corpus_dir': kwargs['gigaspeech']['corpus_dir'],
            'output_dir': kwargs['gigaspeech']['output_dir'],
            'feats_dir': kwargs['gigaspeech']['feats_dir'],
            'batch_duration': kwargs['gigaspeech']['batch_duration'],
        }

        train_cuts = create_gigaspeech_cut(**data_dict, phase='XL', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        dev_cuts = create_gigaspeech_cut(**data_dict, phase='DEV', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        test_cuts = create_gigaspeech_cut(**data_dict, phase='TEST', prepare_dataset=prepare_dataset, get_cuts=get_cuts)

    elif kwargs['dataset_names'][0] == 'voxceleb':

        data_dict = {
            'voxceleb1_root': kwargs['voxceleb']['voxceleb1_root'],
            'voxceleb2_root': kwargs['voxceleb']['voxceleb2_root'],
            'output_dir': kwargs['voxceleb']['output_dir'],
            'feats_dir': kwargs['voxceleb']['feats_dir'],
            'batch_duration': kwargs['voxceleb']['batch_duration'],
        }

        train_cuts = create_voxceleb_cut(**data_dict, phase='train', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        dev_cuts = create_voxceleb_cut(**data_dict, phase='dev', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
        test_cuts = create_voxceleb_cut(**data_dict, phase='test', prepare_dataset=prepare_dataset, get_cuts=get_cuts)
