import ml_collections


def load_config():

    cfg = ml_collections.ConfigDict()

    cfg.task = "run"  # ['wer', 'download', 'prepare', 'compute_feats', 'cuts_subset', 'test_data', 'run']
    cfg.function = "predict"  # ['train', 'test', 'predict', 'predict_sincnet', 'predict_dihard3', 'other_vad'] - if task is 'run'

    cfg.seed = 42
    cfg.device = "gpu"  # ['cpu', 'gpu']
    cfg.num_devices = 1
    cfg.distributed_training = False

    cfg.feature_extractor = (
        "wav2vec2"  # [sincnet, fbank, wav2vec2, hubert_base_robust_mgr]
    )
    feature_extractor_to_append_filename = (
        "window" if cfg.feature_extractor == "sincnet" else cfg.feature_extractor
    )
    cfg.frame_shift = 0.01 if cfg.feature_extractor == "fbank" else 0.02

    # models
    cfg.supported_models = ["PyanNet", "PyanNet2"]
    cfg.model_name = "PyanNet" if cfg.feature_extractor == "sincnet" else "PyanNet2"

    # pyannet
    cfg.model_dict = ml_collections.ConfigDict()
    if cfg.feature_extractor == "sincnet":
        cfg.model_dict.encoding_dim = 60
    elif cfg.feature_extractor == "fbank":
        cfg.model_dict.encoding_dim = 80
    else:
        cfg.model_dict.encoding_dim = 768

    # training
    cfg.max_epochs = 20
    cfg.check_val_every_n_epoch = 3
    cfg.learning_rate = 1e-3
    cfg.batch_size = 80

    # checkpointing
    cfg.experiments_dir = "/export/c01/ashah108/vad/experiments/all-50-wav2vec2-no-musan-weighted-stop-early-false"
    cfg.load_checkpoint = False
    cfg.checkpoint_path = "/export/c01/ashah108/vad/experiments/all-50-wav2vec2-no-musan-weighted-stop-early-false/checkpoint-epoch=11.ckpt"

    # logging
    cfg.is_wandb = False

    # weights and biases (wandb)
    cfg.wandb = ml_collections.ConfigDict()

    cfg.wandb.project = "universal-vad"
    cfg.wandb.name = "all-50-wav2vec2-no-musan-weighted-stop-early-false"

    # datasets
    cfg.supported_datasets = [
        "musan",
        "ami",
        "dihard3",
        "switchboard",
        "eval2000",
        "chime6",
        "tedlium",
        "voxconverse",
        "dipco",
        "gale_arabic",
        "gale_mandarin",
        "santa_barbara",
        "aishell",
        "babel",
        "callhome_english",
        "fisher_english",
        "gigaspeech",
        "mgb2",
    ]
    cfg.prepare_dataset = False
    cfg.get_cuts = False

    # subset size in hours
    cfg.cut_subset = True
    cfg.cut_subset_size = 50

    cfg.enable_musan = True

    subset_and_noise_to_append_filename = ""
    if cfg.cut_subset:
        subset_and_noise_to_append_filename += f"_subset_{cfg.cut_subset_size}"
    if cfg.enable_musan:
        subset_and_noise_to_append_filename += f"_with_noise"

    cfg.max_duration = 400
    cfg.stop_early = False

    # cfg.dataset_names = ["switchboard"]
    cfg.dataset_names = [
        "ami",
        "tedlium",
        "switchboard",
        "voxconverse",
        "chime6",
        "dihard3",
        "gale_arabic",
        "gale_mandarin",
        "callhome_english",
    ]

    # cfg.dataset_weights = {'ami': 70, 'tedlium': 540, 'switchboard': 387, 'voxconverse': 20, 'chime6': 40, 'dihard3': 34, 'dipco': 3, 'gale_arabic': 960, 'gale_mandarin': 821}
    cfg.dataset_weights = {
        "ami": 50,
        "tedlium": 50,
        "switchboard": 50,
        "voxconverse": 20,
        "chime6": 40,
        "dihard3": 34,
        "dipco": 3,
        "gale_arabic": 50,
        "gale_mandarin": 50,
        "callhome_english": 50,
        "gigaspeech": 50,
    }
    # cfg.dataset_weights = None

    # cfg.test_dataset_names = ["dipco", "santa_barbara"]
    cfg.test_dataset_names = [
        "tedlium",
        "chime6",
        "voxconverse",
        "ami",
        "switchboard",
        "gale_arabic",
        "gale_mandarin",
        "dihard3",
        "dipco",
        "santa_barbara",
        "callhome_english",
        "gigaspeech",
    ]

    # cfg.predict_dataset_names = ["dipco"]
    cfg.predict_dataset_names = [
        "tedlium",
        "chime6",
        "voxconverse",
        "ami",
        "switchboard",
        "gale_arabic",
        "gale_mandarin",
        "dihard3",
        "dipco",
        "santa_barbara",
        "callhome_english",
        "gigaspeech",
    ]

    cfg.predict_output_dir = "/export/c01/ashah108/vad/data/predictions"

    # compute feats
    cfg.compute_feats = ml_collections.ConfigDict()

    cfg.compute_feats.cutset_path = "/export/c01/ashah108/vad/data/predictions/train_cuts_tedlium_buffer-0_split.jsonl.gz"
    cfg.compute_feats.new_cuts_filename = "tedlium_cuts_train_0_fbank_split.jsonl.gz"
    cfg.compute_feats.new_feats_filename = "tedlium_cuts_train_0_fbank_split_feats"
    cfg.compute_feats.output_dir = "/export/c01/ashah108/vad/data/tedlium/manifests"
    cfg.compute_feats.feats_dir = "/export/c01/ashah108/vad/data/tedlium/feats"
    cfg.compute_feats.batch_duration = 600

    # santa_barbara
    # added compute_energy() and mix() functions in ssl feature extractor
    cfg.santa_barbara = ml_collections.ConfigDict()
    cfg.santa_barbara.train_cut_set_path = f""
    cfg.santa_barbara.dev_cut_set_path = f""
    cfg.santa_barbara.test_cut_set_path = f"/export/c01/ashah108/vad/data/santa_barbara/manifests/santa_barbara_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.santa_barbara.output_dir = (
        "/export/c01/ashah108/vad/data/santa_barbara/manifests"
    )
    cfg.santa_barbara.feats_dir = "/export/c01/ashah108/vad/data/santa_barbara/feats"
    cfg.santa_barbara.batch_duration = 600

    # musan
    cfg.musan = ml_collections.ConfigDict()
    cfg.musan.cut_set_path = (
        f"/export/c01/ashah108/vad/data/musan/manifests/musan_cuts_window.jsonl.gz"
    )
    cfg.musan.corpus_dir = "/export/corpora6/musan"
    cfg.musan.output_dir = "/export/c01/ashah108/vad/data/musan/manifests"
    cfg.musan.feats_dir = "/export/c01/ashah108/vad/data/musan/feats"
    cfg.musan.batch_duration = 600

    # ami
    cfg.ami = ml_collections.ConfigDict()
    cfg.ami.train_cut_set_path = f"/export/c01/ashah108/vad/data/ami/manifests/train_ihm_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    cfg.ami.dev_cut_set_path = f"/export/c01/ashah108/vad/data/ami/manifests/dev_ihm_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.ami.test_cut_set_path = f"/export/c01/ashah108/vad/data/ami/manifests/test_ihm_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.ami.audio_dir = "/export/corpora5/amicorpus"
    cfg.ami.output_dir = "/export/c01/ashah108/vad/data/ami/manifests"
    cfg.ami.feats_dir = "/export/c01/ashah108/vad/data/ami/feats"
    cfg.ami.mic = "ihm"
    cfg.ami.batch_duration = 600

    # dihard3
    cfg.dihard3 = ml_collections.ConfigDict()
    cfg.dihard3.dev_cut_set_path = f"/export/c01/ashah108/vad/data/dihard3/manifests/dev_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    cfg.dihard3.test_cut_set_path = f"/export/c01/ashah108/vad/data/dihard3/manifests/eval_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.dihard3.dev_audio_dir = "/export/corpora5/LDC/LDC2020E12/LDC2020E12_Third_DIHARD_Challenge_Development_Data"
    cfg.dihard3.eval_audio_dir = "/export/corpora5/LDC/LDC2021E02_Third_DIHARD_Challenge_Evaluation_Data_Complete"
    cfg.dihard3.output_dir = "/export/c01/ashah108/vad/data/dihard3/manifests"
    cfg.dihard3.feats_dir = "/export/c01/ashah108/vad/data/dihard3/feats"
    cfg.dihard3.batch_duration = 600

    # switchboard
    cfg.switchboard = ml_collections.ConfigDict()
    cfg.switchboard.train_cut_set_path = f"/export/c01/ashah108/vad/data/switchboard/manifests/train_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    cfg.switchboard.dev_cut_set_path = f"/export/c01/ashah108/vad/data/switchboard/manifests/dev_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.switchboard.test_cut_set_path = f"/export/c01/ashah108/vad/data/eval2000/manifests/test_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.switchboard.audio_dir = "/export/corpora3/LDC/LDC97S62/swb1"
    cfg.switchboard.transcripts_dir = "/export/c01/ashah108/vad/swb_ms98_transcriptions"
    cfg.switchboard.output_dir = "/export/c01/ashah108/vad/data/switchboard/manifests"
    cfg.switchboard.feats_dir = "/export/c01/ashah108/vad/data/switchboard/feats"
    cfg.switchboard.batch_duration = 600

    # eval2000
    cfg.eval2000 = ml_collections.ConfigDict()
    cfg.eval2000.test_cut_set_path = f"/export/c01/ashah108/vad/data/eval2000/manifests/test_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.eval2000.corpus_dir = "/export/corpora5/LDC"
    cfg.eval2000.output_dir = "/export/c01/ashah108/vad/data/eval2000/manifests"
    cfg.eval2000.feats_dir = "/export/c01/ashah108/vad/data/eval2000/feats"
    cfg.eval2000.batch_duration = 600

    # callhome_english
    cfg.callhome_english = ml_collections.ConfigDict()
    cfg.callhome_english.train_cut_set_path = f"/export/c01/ashah108/vad/data/callhome_english/manifests/train_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    cfg.callhome_english.dev_cut_set_path = f"/export/c01/ashah108/vad/data/callhome_english/manifests/devtest_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.callhome_english.test_cut_set_path = f"/export/c01/ashah108/vad/data/callhome_english/manifests/evaltest_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.callhome_english.audio_dir = "/export/corpora5/LDC/LDC97S42"
    cfg.callhome_english.transcript_dir = "/export/corpora5/LDC/LDC97T14"
    cfg.callhome_english.output_dir = (
        "/export/c01/ashah108/vad/data/callhome_english/manifests"
    )
    cfg.callhome_english.feats_dir = (
        "/export/c01/ashah108/vad/data/callhome_english/feats"
    )
    cfg.callhome_english.batch_duration = 600

    # # callhome_egyptian
    # # changed path in lhotse recipe for transcripts
    # cfg.callhome_egyptian = ml_collections.ConfigDict()
    # cfg.callhome_egyptian.train_cut_set_path = f'/export/c01/ashah108/vad/data/callhome_egyptian/manifests/train_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz'
    # cfg.callhome_egyptian.dev_cut_set_path = f'/export/c01/ashah108/vad/data/callhome_egyptian/manifests/devtest_cuts_{feature_extractor_to_append_filename}.jsonl.gz'
    # cfg.callhome_egyptian.test_cut_set_path = f'/export/c01/ashah108/vad/data/callhome_egyptian/manifests/evaltest_cuts_{feature_extractor_to_append_filename}.jsonl.gz'
    # cfg.callhome_egyptian.audio_dir = '/export/corpora5/LDC/LDC97S45'
    # cfg.callhome_egyptian.transcript_dir = '/export/corpora5/LDC/LDC97T19'
    # cfg.callhome_egyptian.output_dir = '/export/c01/ashah108/vad/data/callhome_egyptian/manifests'
    # cfg.callhome_egyptian.feats_dir = '/export/c01/ashah108/vad/data/callhome_egyptian/feats'
    # cfg.callhome_egyptian.batch_duration = 600

    # chime6
    cfg.chime6 = ml_collections.ConfigDict()
    cfg.chime6.train_cut_set_path = f"/export/c01/ashah108/vad/data/chime6/manifests/train_mdm_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    cfg.chime6.dev_cut_set_path = f"/export/c01/ashah108/vad/data/chime6/manifests/dev_mdm_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.chime6.test_cut_set_path = f"/export/c01/ashah108/vad/data/chime6/manifests/eval_mdm_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.chime6.corpus_dir = (
        "/export/c01/draj/kaldi_chime6_jhu/egs/chime6/s5_track2/CHiME6"
    )
    cfg.chime6.output_dir = "/export/c01/ashah108/vad/data/chime6/manifests"
    cfg.chime6.mic = "mdm"  # ihm doesn't work - no audio files in eval set for P*
    cfg.chime6.feats_dir = "/export/c01/ashah108/vad/data/chime6/feats"
    cfg.chime6.batch_duration = 600

    # gigaspeech
    # added check to continue if len of sups is 0 in lhotse recipe
    cfg.gigaspeech = ml_collections.ConfigDict()
    cfg.gigaspeech.train_cut_set_path = f"/export/c01/ashah108/vad/data/gigaspeech/manifests/train_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    cfg.gigaspeech.dev_cut_set_path = f"/export/c01/ashah108/vad/data/gigaspeech/manifests/dev_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.gigaspeech.test_cut_set_path = f"/export/c01/ashah108/vad/data/gigaspeech/manifests/test_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.gigaspeech.corpus_dir = "/export/corpora5/gigaspeech"
    cfg.gigaspeech.output_dir = "/export/c01/ashah108/vad/data/gigaspeech/manifests"
    cfg.gigaspeech.feats_dir = "/export/c01/ashah108/vad/data/gigaspeech/feats"
    cfg.gigaspeech.batch_duration = 600

    # # fisher_english
    # # added condition in lhotse for audio dirs (skip if sub dir is "audio" as it accesses fe_0x_px_sphx/audio and CLSP has another audio subdir as well)
    # cfg.fisher_english = ml_collections.ConfigDict()
    # cfg.fisher_english.train_cut_set_path = f"/export/c01/ashah108/vad/data/fisher_english/manifests/train_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    # cfg.fisher_english.dev_cut_set_path = f"/export/c01/ashah108/vad/data/fisher_english/manifests/devtest_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    # cfg.fisher_english.test_cut_set_path = f"/export/c01/ashah108/vad/data/fisher_english/manifests/evaltest_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    # cfg.fisher_english.corpus_dir = "/export/corpora3/LDC"
    # cfg.fisher_english.audio_dirs = ["LDC2004S13", "LDC2005S13"]
    # cfg.fisher_english.transcript_dirs = [
    #     "LDC2004T19/fe_03_p1_tran",
    #     "LDC2005T19/fe_03_p2_tran",
    # ]
    # cfg.fisher_english.output_dir = (
    #     "/export/c01/ashah108/vad/data/fisher_english/manifests"
    # )
    # cfg.fisher_english.feats_dir = "/export/c01/ashah108/vad/data/fisher_english/feats"
    # cfg.fisher_english.batch_duration = 600

    # tedlium
    # icefall - convert_text_to_ids() in compute_loss() has multiple crguments of sp - remove unk_id
    cfg.tedlium = ml_collections.ConfigDict()
    cfg.tedlium.train_cut_set_path = f"/export/c01/ashah108/vad/data/tedlium/manifests/train_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    cfg.tedlium.dev_cut_set_path = f"/export/c01/ashah108/vad/data/tedlium/manifests/dev_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.tedlium.test_cut_set_path = f"/export/c01/ashah108/vad/data/tedlium/manifests/test_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.tedlium.tedlium_root = "/export/corpora5/TEDLIUM_release-3"
    cfg.tedlium.output_dir = "/export/c01/ashah108/vad/data/tedlium/manifests"
    cfg.tedlium.feats_dir = "/export/c01/ashah108/vad/data/tedlium/feats"
    cfg.tedlium.batch_duration = 600

    # voxconverse
    # modify to add paths separately for wav and rttm files
    cfg.voxconverse = ml_collections.ConfigDict()
    cfg.voxconverse.train_cut_set_path = f"/export/c01/ashah108/vad/data/voxconverse/manifests/train_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    cfg.voxconverse.dev_cut_set_path = f"/export/c01/ashah108/vad/data/voxconverse/manifests/dev_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.voxconverse.test_cut_set_path = f"/export/c01/ashah108/vad/data/voxconverse/manifests/test_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.voxconverse.corpus_dir = "/export/corpora5/voxconverse"
    cfg.voxconverse.output_dir = "/export/c01/ashah108/vad/data/voxconverse/manifests"
    cfg.voxconverse.feats_dir = "/export/c01/ashah108/vad/data/voxconverse/feats"
    cfg.voxconverse.batch_duration = 600

    # dipco
    # changed CORPUS_URL to 'https://zenodo.org/records/8122551/files/DipCo.tgz?download=1' in lhotse recipe for download
    cfg.dipco = ml_collections.ConfigDict()
    cfg.dipco.target_dir = "/export/c01/ashah108"
    cfg.dipco.train_cut_set_path = f"/export/c01/ashah108/vad/data/dipco/manifests/train_ihm_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    cfg.dipco.dev_cut_set_path = f"/export/c01/ashah108/vad/data/dipco/manifests/dev_ihm_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.dipco.test_cut_set_path = f"/export/c01/ashah108/vad/data/dipco/manifests/eval_ihm_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.dipco.corpus_dir = "/export/c01/ashah108/Dipco"
    cfg.dipco.output_dir = "/export/c01/ashah108/vad/data/dipco/manifests"
    cfg.dipco.mic = "ihm"
    cfg.dipco.feats_dir = "/export/c01/ashah108/vad/data/dipco/feats"
    cfg.dipco.batch_duration = 600

    # # voxceleb
    # # changed "," to "\t" (or removed ",") in _prepare_voxceleb_v2() as csv file was tab separated
    # cfg.voxceleb = ml_collections.ConfigDict()
    # cfg.voxceleb.train_cut_set_path = f'/export/c01/ashah108/vad/data/voxceleb/manifests/train_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz'
    # cfg.voxceleb.dev_cut_set_path = f'/export/c01/ashah108/vad/data/voxceleb/manifests/dev_cuts_{feature_extractor_to_append_filename}.jsonl.gz'
    # cfg.voxceleb.test_cut_set_path = f'/export/c01/ashah108/vad/data/voxceleb/manifests/test_cuts_{feature_extractor_to_append_filename}.jsonl.gz'
    # cfg.voxceleb.voxceleb1_root = '/export/corpora5/VoxCeleb1_v2'
    # cfg.voxceleb.voxceleb2_root = '/export/corpora5/VoxCeleb2'
    # cfg.voxceleb.output_dir = '/export/c01/ashah108/vad/data/voxceleb/manifests'
    # cfg.voxceleb.feats_dir = '/export/c01/ashah108/vad/data/voxceleb/feats'
    # cfg.voxceleb.batch_duration = 600

    # # mgb2
    # # download_mgb2 not imported in __init__.py in lhotse
    # cfg.mgb2 = ml_collections.ConfigDict()
    # cfg.mgb2.train_cut_set_path = f"/export/c01/ashah108/vad/data/mgb2/manifests/train_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    # cfg.mgb2.dev_cut_set_path = f"/export/c01/ashah108/vad/data/mgb2/manifests/dev_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    # cfg.mgb2.test_cut_set_path = f"/export/c01/ashah108/vad/data/mgb2/manifests/test_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    # cfg.mgb2.corpus_dir = "/export/corpora5/MGB/MGB-2"
    # cfg.mgb2.output_dir = "/export/c01/ashah108/vad/data/mgb2/manifests"
    # cfg.mgb2.feats_dir = "/export/c01/ashah108/vad/data/mgb2/feats"
    # cfg.mgb2.batch_duration = 600

    # gale_arabic
    # remove santa_barbara properties, text, speaker, gender in supervision segment due to json decode error (NaN values)
    cfg.gale_arabic = ml_collections.ConfigDict()
    cfg.gale_arabic.train_cut_set_path = f"/export/c01/ashah108/vad/data/gale_arabic/manifests/train_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    cfg.gale_arabic.dev_cut_set_path = f"/export/c01/ashah108/vad/data/gale_arabic/manifests/dev_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.gale_arabic.test_cut_set_path = f"/export/c01/ashah108/vad/data/gale_arabic/manifests/test_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.gale_arabic.audio_dirs = [
        "/export/corpora5/LDC/LDC2013S02",
        "/export/corpora5/LDC/LDC2013S07",
        "/export/corpora5/LDC/LDC2014S07",
        "/export/corpora5/LDC/LDC2015S01",
        "/export/corpora5/LDC/LDC2015S11",
        "/export/corpora5/LDC/LDC2016S01",
    ]
    # '/export/corpora5/LDC/LDC2016S07', '/export/corpora3/LDC/LDC2017S02',
    # '/export/corpora3/LDC/LDC2017S15', '/export/corpora3/LDC/LDC2018S05']
    cfg.gale_arabic.transcript_dirs = [
        "/export/corpora5/LDC/LDC2013T17",
        "/export/corpora5/LDC/LDC2013T04",
        "/export/corpora5/LDC/LDC2014T17",
        "/export/corpora5/LDC/LDC2015T01",
        "/export/corpora5/LDC/LDC2015T16",
        "/export/corpora5/LDC/LDC2016T06",
    ]
    # '/export/corpora5/LDC/LDC2016T17', '/export/corpora3/LDC/LDC2017T04',
    # '/export/corpora3/LDC/LDC2017T12', '/export/corpora3/LDC/LDC2018T14']
    cfg.gale_arabic.output_dir = "/export/c01/ashah108/vad/data/gale_arabic/manifests"
    cfg.gale_arabic.feats_dir = "/export/c01/ashah108/vad/data/gale_arabic/feats"
    cfg.gale_arabic.batch_duration = 600

    # gale_arabic
    # change kaldi URL to raw + tree/master in lhotse recipe
    # remove santa_barbara properties, text, speaker, gender in supervision segment due to json decode error (NaN values)
    cfg.gale_mandarin = ml_collections.ConfigDict()
    cfg.gale_mandarin.train_cut_set_path = f"/export/c01/ashah108/vad/data/gale_mandarin/manifests/train_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    cfg.gale_mandarin.dev_cut_set_path = f"/export/c01/ashah108/vad/data/gale_mandarin/manifests/dev_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.gale_mandarin.test_cut_set_path = f"/export/c01/ashah108/vad/data/gale_mandarin/manifests/test_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    cfg.gale_mandarin.audio_dirs = [
        "/export/corpora5/LDC/LDC2013S08",
        "/export/corpora5/LDC/LDC2013S04",
        "/export/corpora5/LDC/LDC2014S09",
        "/export/corpora5/LDC/LDC2015S06",
        "/export/corpora5/LDC/LDC2015S13",
        "/export/corpora5/LDC/LDC2016S03",
    ]
    cfg.gale_mandarin.transcript_dirs = [
        "/export/corpora5/LDC/LDC2013T20",
        "/export/corpora5/LDC/LDC2013T08",
        "/export/corpora5/LDC/LDC2014T28",
        "/export/corpora5/LDC/LDC2015T09",
        "/export/corpora5/LDC/LDC2015T25",
        "/export/corpora5/LDC/LDC2016T12",
    ]
    cfg.gale_mandarin.output_dir = (
        "/export/c01/ashah108/vad/data/gale_mandarin/manifests"
    )
    cfg.gale_mandarin.feats_dir = "/export/c01/ashah108/vad/data/gale_mandarin/feats"
    cfg.gale_mandarin.batch_duration = 600

    # # babel
    # cfg.babel = ml_collections.ConfigDict()
    # cfg.babel.train_cut_set_path = f"/export/c01/ashah108/vad/data/babel/manifests/train_cuts_{feature_extractor_to_append_filename}{subset_and_noise_to_append_filename}.jsonl.gz"
    # cfg.babel.dev_cut_set_path = f"/export/c01/ashah108/vad/data/babel/manifests/dev_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    # cfg.babel.test_cut_set_path = f"/export/c01/ashah108/vad/data/babel/manifests/test_cuts_{feature_extractor_to_append_filename}.jsonl.gz"
    # cfg.babel.corpus_dir = "/export/corpora5/babel_CLSP"
    # cfg.babel.relative_lang_paths = [
    #     "101-cantonese/release-current",
    #     "102-assamese/release-current",
    #     "103-bengali/release-current",
    #     "105-turkish/release-current",
    #     "106-tagalog/release-current",
    #     "107-vietnamese/release-current",
    #     "201-haitian/release-current",
    #     "203-lao/release-current",
    #     "204-tamil/release-current",
    # ]
    # cfg.babel.output_dir = "/export/c01/ashah108/vad/data/babel/manifests"
    # cfg.babel.feats_dir = "/export/c01/ashah108/vad/data/babel/feats"
    # cfg.babel.batch_duration = 600

    return cfg
