
import ml_collections


def load_config():

    cfg = ml_collections.ConfigDict()

    cfg.seed = 42
    cfg.device = 'gpu'  # ['cpu', 'gpu'] cuda??

    # models
    cfg.supported_models = ['PyanNet']
    cfg.model_name = 'PyanNet'

    # pyannet
    cfg.model_dict = ml_collections.ConfigDict()

    # training
    cfg.max_epochs = 1
    cfg.learning_rate = 1e-3
    cfg.batch_size = 60

    # checkpointing
    cfg.experiments_dir = '/export/c01/ashah108/vad/experiments/lbrsp-pyannet'
    cfg.from_checkpoint = False
    cfg.checkpoint_path = '/export/c01/ashah108/vad/experiments/lbrsp-pyannet'

    # logging
    cfg.is_wandb = False

    # weights and biases (wandb)
    cfg.wandb = ml_collections.ConfigDict()
    
    cfg.wandb.project = 'universal-vad'
    cfg.wandb.run = 'librispeech-pyannet-fbank'

    # datasets
    cfg.supported_datasets = ['librispeech']
    cfg.dataset_name = 'librispeech'

    # librispeech
    cfg.librispeech = ml_collections.ConfigDict()

    cfg.librispeech.train_cut_set_path = '/export/c01/ashah108/vad/data/librispeech/manifests/train_360_cuts_feats.jsonl.gz'
    cfg.librispeech.dev_cut_set_path = '/export/c01/ashah108/vad/data/librispeech/manifests/dev_cuts_feats.jsonl.gz'
    cfg.librispeech.test_cut_set_path = '/export/c01/ashah108/vad/data/librispeech/manifests/test_cuts_feats.jsonl.gz'
    cfg.librispeech.audio_dir = '/export/corpora5/LibriSpeech'
    cfg.librispeech.output_dir = '/export/c01/ashah108/vad/data/librispeech/manifests'
    cfg.librispeech.feats_dir = '/export/c01/ashah108/vad/data/librispeech/feats'
    cfg.librispeech.batch_duration = 600
    cfg.librispeech.max_duration = 600

    return cfg
