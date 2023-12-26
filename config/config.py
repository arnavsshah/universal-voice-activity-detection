
import ml_collections


def load_config():

    cfg = ml_collections.ConfigDict()

    cfg.task = 'run'  # ['prepare', 'test_data', 'run']

    cfg.seed = 42
    cfg.device = 'gpu'  # ['cpu', 'gpu'] cuda??

    cfg.function = 'test'  # ['train', 'test', 'predict']
    
    # models
    cfg.supported_models = ['PyanNet']
    cfg.model_name = 'PyanNet'

    # pyannet
    cfg.model_dict = ml_collections.ConfigDict()

    cfg.model_dict.encoding_dim = 768  # [768, 80]

    # training
    cfg.max_epochs = 12
    cfg.learning_rate = 1e-3
    cfg.batch_size = 80

    # checkpointing
    cfg.experiments_dir = '/export/c01/ashah108/vad/experiments/ami-ted-swbd-ssl-pyannet'
    cfg.load_checkpoint = True
    cfg.checkpoint_path = '/export/c01/ashah108/vad/experiments/ami-ted-swbd-ssl-pyannet/checkpoint-epoch=10.ckpt'

    # logging
    cfg.is_wandb = False

    # weights and biases (wandb)
    cfg.wandb = ml_collections.ConfigDict()
    
    cfg.wandb.project = 'universal-vad'
    cfg.wandb.name = 'ami-ted-swbd-ssl-pyannet'

    # datasets
    cfg.supported_datasets = ['ami', 'dihard3', 'switchboard', 'callhome_english', 'callhome_egyptian', 'chime6', 'fisher_english', 'tedlium', 'voxconverse']
    cfg.prepare_dataset = False
    cfg.get_cuts = False
    cfg.dataset_names = ['ami', 'tedlium']
    cfg.dataset_weights = {'ami': 70, 'tedlium': 540}
    # cfg.dataset_weights = {'ami': 70, 'dihard3': 34, 'switchboard': 181, 'chime6': 40, 'tedlium': 540, 'voxconverse': 20}
    cfg.test_dataset_names = ['ami', 'switchboard', 'chime6', 'tedlium', 'voxconverse', 'dihard3']
    cfg.predict_dataset_names = ['ami']  # do only 1 for now
    cfg.max_duration = 400
    cfg.stop_early = True

    cfg.predict_output_dir = '/export/c01/ashah108/vad/data/predictions'

    # librispeech
    cfg.librispeech = ml_collections.ConfigDict()

    cfg.librispeech.train_cut_set_path = '/export/c01/ashah108/vad/data/librispeech/manifests/train_360_cuts_feats_ssl.jsonl.gz'
    cfg.librispeech.dev_cut_set_path = '/export/c01/ashah108/vad/data/librispeech/manifests/dev_cuts_feats_ssl.jsonl.gz'
    cfg.librispeech.test_cut_set_path = '/export/c01/ashah108/vad/data/librispeech/manifests/test_cuts_feats_ssl.jsonl.gz'
    cfg.librispeech.audio_dir = ''
    cfg.librispeech.output_dir = '/export/c01/ashah108/vad/data/librispeech/manifests'
    cfg.librispeech.feats_dir = '/export/c01/ashah108/vad/data/librispeech/feats'
    cfg.librispeech.batch_duration = 600

    # ami
    cfg.ami = ml_collections.ConfigDict()

    cfg.ami.train_cut_set_path = '/export/c01/ashah108/vad/data/ami/manifests/train_ihm_cuts_feats_ssl.jsonl.gz'
    cfg.ami.dev_cut_set_path = '/export/c01/ashah108/vad/data/ami/manifests/dev_ihm_cuts_feats_ssl.jsonl.gz'
    cfg.ami.test_cut_set_path = '/export/c01/ashah108/vad/data/ami/manifests/test_ihm_cuts_feats_ssl.jsonl.gz'
    cfg.ami.audio_dir = '/export/corpora5/amicorpus'
    cfg.ami.output_dir = '/export/c01/ashah108/vad/data/ami/manifests'
    cfg.ami.feats_dir = '/export/c01/ashah108/vad/data/ami/feats'
    cfg.ami.mic = 'ihm'
    cfg.ami.batch_duration = 600

    # dihard3
    cfg.dihard3 = ml_collections.ConfigDict()

    cfg.dihard3.dev_cut_set_path = '/export/c01/ashah108/vad/data/dihard3/manifests/dev_cuts_feats_ssl.jsonl.gz'
    cfg.dihard3.test_cut_set_path = '/export/c01/ashah108/vad/data/dihard3/manifests/eval_cuts_feats_ssl.jsonl.gz'
    cfg.dihard3.dev_audio_dir = '/export/corpora5/LDC/LDC2020E12/LDC2020E12_Third_DIHARD_Challenge_Development_Data'
    cfg.dihard3.eval_audio_dir = '/export/corpora5/LDC/LDC2021E02_Third_DIHARD_Challenge_Evaluation_Data_Complete'
    cfg.dihard3.output_dir = '/export/c01/ashah108/vad/data/dihard3/manifests'
    cfg.dihard3.feats_dir = '/export/c01/ashah108/vad/data/dihard3/feats'
    cfg.dihard3.batch_duration = 600

    # switchboard
    cfg.switchboard = ml_collections.ConfigDict()

    cfg.switchboard.train_cut_set_path = '/export/c01/ashah108/vad/data/switchboard/manifests/swbd_train_cuts_feats_ssl.jsonl.gz'
    cfg.switchboard.dev_cut_set_path = '/export/c01/ashah108/vad/data/switchboard/manifests/swbd_dev_cuts_feats_ssl.jsonl.gz'
    cfg.switchboard.test_cut_set_path = '/export/c01/ashah108/vad/data/switchboard/manifests/swbd_test_cuts_feats_ssl.jsonl.gz'
    cfg.switchboard.audio_dir = '/export/corpora3/LDC/LDC97S62/swb1'
    cfg.switchboard.transcripts_dir = '/export/c01/ashah108/vad/swb_ms98_transcriptions'
    cfg.switchboard.output_dir = '/export/c01/ashah108/vad/data/switchboard/manifests'
    cfg.switchboard.feats_dir = '/export/c01/ashah108/vad/data/switchboard/feats'
    cfg.switchboard.batch_duration = 600

    # callhome_english
    cfg.callhome_english = ml_collections.ConfigDict()

    cfg.callhome_english.train_cut_set_path = '/export/c01/ashah108/vad/data/callhome_english/manifests/train_cuts_feats_ssl.jsonl.gz'
    cfg.callhome_english.dev_cut_set_path = '/export/c01/ashah108/vad/data/callhome_english/manifests/devtest_cuts_feats_ssl.jsonl.gz'
    cfg.callhome_english.test_cut_set_path = '/export/c01/ashah108/vad/data/callhome_english/manifests/evaltest_cuts_feats_ssl.jsonl.gz'
    cfg.callhome_english.audio_dir = '/export/corpora5/LDC/LDC97S42'
    cfg.callhome_english.transcript_dir = '/export/corpora5/LDC/LDC97T14'
    cfg.callhome_english.output_dir = '/export/c01/ashah108/vad/data/callhome_english/manifests'
    cfg.callhome_english.feats_dir = '/export/c01/ashah108/vad/data/callhome_english/feats'
    cfg.callhome_english.batch_duration = 600

    # callhome_egyptian
    # changed path in lhotse recipe for transcripts
    cfg.callhome_egyptian = ml_collections.ConfigDict()

    cfg.callhome_egyptian.train_cut_set_path = '/export/c01/ashah108/vad/data/callhome_egyptian/manifests/train_cuts_feats_ssl.jsonl.gz'
    cfg.callhome_egyptian.dev_cut_set_path = '/export/c01/ashah108/vad/data/callhome_egyptian/manifests/devtest_cuts_feats_ssl.jsonl.gz'
    cfg.callhome_egyptian.test_cut_set_path = '/export/c01/ashah108/vad/data/callhome_egyptian/manifests/evaltest_cuts_feats_ssl.jsonl.gz'
    cfg.callhome_egyptian.audio_dir = '/export/corpora5/LDC/LDC97S45'
    cfg.callhome_egyptian.transcript_dir = '/export/corpora5/LDC/LDC97T19'
    cfg.callhome_egyptian.output_dir = '/export/c01/ashah108/vad/data/callhome_egyptian/manifests'
    cfg.callhome_egyptian.feats_dir = '/export/c01/ashah108/vad/data/callhome_egyptian/feats'
    cfg.callhome_egyptian.batch_duration = 600

    # chime6
    cfg.chime6 = ml_collections.ConfigDict()

    cfg.chime6.train_cut_set_path = '/export/c01/ashah108/vad/data/chime6/manifests/train_mdm_cuts_feats_ssl.jsonl.gz'
    cfg.chime6.dev_cut_set_path = '/export/c01/ashah108/vad/data/chime6/manifests/dev_mdm_cuts_feats_ssl.jsonl.gz'
    cfg.chime6.test_cut_set_path = '/export/c01/ashah108/vad/data/chime6/manifests/eval_mdm_cuts_feats_ssl.jsonl.gz'
    cfg.chime6.corpus_dir = '/export/c01/draj/kaldi_chime6_jhu/egs/chime6/s5_track2/CHiME6'
    cfg.chime6.output_dir = '/export/c01/ashah108/vad/data/chime6/manifests'
    cfg.chime6.mic = 'mdm'  # ihm doesn't work - no audio files in eval set for P*
    cfg.chime6.feats_dir = '/export/c01/ashah108/vad/data/chime6/feats'
    cfg.chime6.batch_duration = 600

    # # gigaspeech
    # cfg.gigaspeech = ml_collections.ConfigDict()

    # cfg.gigaspeech.train_cut_set_path = '/export/c01/ashah108/vad/data/gigaspeech/manifests/XL_cuts_feats_ssl.jsonl.gz'
    # cfg.gigaspeech.dev_cut_set_path = '/export/c01/ashah108/vad/data/gigaspeech/manifests/DEV_cuts_feats_ssl.jsonl.gz'
    # cfg.gigaspeech.test_cut_set_path = '/export/c01/ashah108/vad/data/gigaspeech/manifests/TEST_cuts_feats_ssl.jsonl.gz'
    # cfg.gigaspeech.corpus_dir = '/export/corpora5/gigaspeech'
    # cfg.gigaspeech.output_dir = '/export/c01/ashah108/vad/data/gigaspeech/manifests'
    # cfg.gigaspeech.feats_dir = '/export/c01/ashah108/vad/data/gigaspeech/feats'
    # cfg.gigaspeech.batch_duration = 600

    # # voxceleb
    # # changed "," to "\t" in _prepare_voxceleb_v2() as csv file was tab separated
    # cfg.voxceleb = ml_collections.ConfigDict()

    # cfg.voxceleb.train_cut_set_path = '/export/c01/ashah108/vad/data/voxceleb/manifests/train_cuts_feats_ssl.jsonl.gz'
    # cfg.voxceleb.dev_cut_set_path = '/export/c01/ashah108/vad/data/voxceleb/manifests/dev_cuts_feats_ssl.jsonl.gz'
    # cfg.voxceleb.test_cut_set_path = '/export/c01/ashah108/vad/data/voxceleb/manifests/test_cuts_feats_ssl.jsonl.gz'
    # cfg.voxceleb.voxceleb1_root = '/export/corpora5/VoxCeleb1_v2'
    # cfg.voxceleb.voxceleb2_root = '/export/corpora5/VoxCeleb2'
    # cfg.voxceleb.output_dir = '/export/c01/ashah108/vad/data/voxceleb/manifests'
    # cfg.voxceleb.feats_dir = '/export/c01/ashah108/vad/data/voxceleb/feats'
    # cfg.voxceleb.batch_duration = 600

    # fisher_english
    # added condition in lhotse for audio dirs (skip if sub dir is "audio" as it accesses fe_0x_px_sphx/audio and CLSP has another audio subdir as well)
    cfg.fisher_english = ml_collections.ConfigDict()

    cfg.fisher_english.train_cut_set_path = '/export/c01/ashah108/vad/data/fisher_english/manifests/train_cuts_feats_ssl.jsonl.gz'
    cfg.fisher_english.dev_cut_set_path = '/export/c01/ashah108/vad/data/fisher_english/manifests/devtest_cuts_feats_ssl.jsonl.gz'
    cfg.fisher_english.test_cut_set_path = '/export/c01/ashah108/vad/data/fisher_english/manifests/evaltest_cuts_feats_ssl.jsonl.gz'
    cfg.fisher_english.corpus_dir = '/export/corpora3/LDC'
    cfg.fisher_english.audio_dirs = ['LDC2004S13', 'LDC2005S13']
    cfg.fisher_english.transcript_dirs = ['LDC2004T19/fe_03_p1_tran', 'LDC2005T19/fe_03_p2_tran']
    cfg.fisher_english.output_dir = '/export/c01/ashah108/vad/data/fisher_english/manifests'
    cfg.fisher_english.feats_dir = '/export/c01/ashah108/vad/data/fisher_english/feats'
    cfg.fisher_english.batch_duration = 600

    # tedlium
    cfg.tedlium = ml_collections.ConfigDict()

    cfg.tedlium.train_cut_set_path = '/export/c01/ashah108/vad/data/tedlium/manifests/train_cuts_feats_ssl.jsonl.gz'
    cfg.tedlium.dev_cut_set_path = '/export/c01/ashah108/vad/data/tedlium/manifests/dev_cuts_feats_ssl.jsonl.gz'
    cfg.tedlium.test_cut_set_path = '/export/c01/ashah108/vad/data/tedlium/manifests/test_cuts_feats_ssl.jsonl.gz'
    cfg.tedlium.tedlium_root = '/export/corpora5/TEDLIUM_release-3'
    cfg.tedlium.output_dir = '/export/c01/ashah108/vad/data/tedlium/manifests'
    cfg.tedlium.feats_dir = '/export/c01/ashah108/vad/data/tedlium/feats'
    cfg.tedlium.batch_duration = 600

    # voxconverse
    # modify to add paths separately for wav and rttm files
    cfg.voxconverse = ml_collections.ConfigDict()

    cfg.voxconverse.train_cut_set_path = '/export/c01/ashah108/vad/data/voxconverse/manifests/train_cuts_feats_ssl.jsonl.gz'
    cfg.voxconverse.dev_cut_set_path = '/export/c01/ashah108/vad/data/voxconverse/manifests/dev_cuts_feats_ssl.jsonl.gz'
    cfg.voxconverse.test_cut_set_path = '/export/c01/ashah108/vad/data/voxconverse/manifests/test_cuts_feats_ssl.jsonl.gz'
    cfg.voxconverse.corpus_dir = '/export/corpora5/voxconverse'
    cfg.voxconverse.output_dir = '/export/c01/ashah108/vad/data/voxconverse/manifests'
    cfg.voxconverse.feats_dir = '/export/c01/ashah108/vad/data/voxconverse/feats'
    cfg.voxconverse.batch_duration = 600

    return cfg
