from src.datasets import *
from src.utils.helper import compute_feats_for_cutset, get_cuts_subset
from config.config import *


def prepare_feats(**kwargs):
    assert (
        kwargs["dataset_names"][0] in kwargs["supported_datasets"]
    ), f"Invalid dataset {kwargs['dataset_names'][0]}. Dataset should be one of {kwargs['supported_datasets']}"

    data_dict = {
        "cut_set_path": kwargs["compute_feats"]["cut_set_path"],
        "new_cuts_filename": kwargs["compute_feats"]["new_cuts_filename"],
        "new_feats_filename": kwargs["compute_feats"]["new_feats_filename"],
        "output_dir": kwargs["compute_feats"]["output_dir"],
        "feats_dir": kwargs["compute_feats"]["feats_dir"],
        "batch_duration": kwargs["compute_feats"]["batch_duration"],
    }

    compute_feats_for_cutset(**data_dict)


def prepare_cuts_subset(**kwargs) -> None:

    for dataset_name in kwargs["dataset_names"]:

        assert (
            dataset_name in kwargs["supported_datasets"]
        ), f"Invalid dataset {dataset_name}. Dataset should be one of {kwargs['supported_datasets']}"

        if dataset_name == "dihard3":
            get_cuts_subset(
                kwargs["dihard3"]["dev_cut_set_path"], kwargs["cut_subset_size"]
            )
        else:
            get_cuts_subset(
                kwargs[dataset_name]["train_cut_set_path"], kwargs["cut_subset_size"]
            )


def prepare_data(prepare_dataset=True, get_cuts=True, **kwargs):

    assert (
        kwargs["dataset_names"][0] in kwargs["supported_datasets"]
    ), f"Invalid dataset {kwargs['dataset_names'][0]}. Dataset should be one of {kwargs['supported_datasets']}"

    data_dict = {
        **kwargs[kwargs["dataset_names"][0]],
        "subset": kwargs["cut_subset"],
        "subset_size": kwargs["cut_subset_size"],
        "add_noise": kwargs["enable_musan"],
        "noise_cut_set_path": kwargs["musan"]["cut_set_path"],
        "feature_extractor": kwargs["feature_extractor"],
    }

    if kwargs["dataset_names"][0] == "musan":
        if prepare_dataset:
            create_musan_dataset(**data_dict)

        if get_cuts:
            cuts = create_musan_cut(**data_dict)

    elif kwargs["dataset_names"][0] == "ami":
        if prepare_dataset:
            create_ami_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_ami_cut(**data_dict, phase="train")
            dev_cuts = create_ami_cut(**data_dict, phase="dev")
            test_cuts = create_ami_cut(**data_dict, phase="test")

    elif kwargs["dataset_names"][0] == "dihard3":
        if prepare_dataset:
            create_dihard3_dataset(**data_dict)

        if get_cuts:
            dev_cuts = create_dihard3_cut(**data_dict, phase="dev")
            test_cuts = create_dihard3_cut(**data_dict, phase="eval")

    elif kwargs["dataset_names"][0] == "switchboard":
        if prepare_dataset:
            create_switchboard_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_switchboard_cut(**data_dict, phase="train")
            dev_cuts = create_switchboard_cut(**data_dict, phase="dev")

    elif kwargs["dataset_names"][0] == "eval2000":
        if prepare_dataset:
            create_eval2000_dataset(**data_dict)

        if get_cuts:
            test_cuts = create_eval2000_cut(**data_dict)

    elif kwargs["dataset_names"][0] == "chime6":
        if prepare_dataset:
            create_chime6_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_chime6_cut(**data_dict, phase="train")
            dev_cuts = create_chime6_cut(**data_dict, phase="dev")
            test_cuts = create_chime6_cut(**data_dict, phase="eval")

    elif kwargs["dataset_names"][0] == "tedlium":
        if prepare_dataset:
            create_tedlium_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_tedlium_cut(**data_dict, phase="train")
            dev_cuts = create_tedlium_cut(**data_dict, phase="dev")
            test_cuts = create_tedlium_cut(**data_dict, phase="test")

    elif kwargs["dataset_names"][0] == "voxconverse":
        if prepare_dataset:
            create_voxconverse_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_voxconverse_cut(**data_dict, phase="train")
            dev_cuts = create_voxconverse_cut(**data_dict, phase="dev")
            test_cuts = create_voxconverse_cut(**data_dict, phase="test")

    elif kwargs["dataset_names"][0] == "dipco":
        if prepare_dataset:
            create_dipco_dataset(**data_dict)

        if get_cuts:
            train_cuts, dev_cuts = create_dipco_cut(**data_dict, phase="dev")
            test_cuts, _ = create_dipco_cut(**data_dict, phase="eval")

    elif kwargs["dataset_names"][0] == "gale_arabic":
        if prepare_dataset:
            create_gale_arabic_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_gale_arabic_cut(**data_dict, phase="train")
            dev_cuts = create_gale_arabic_cut(**data_dict, phase="dev")
            test_cuts = create_gale_arabic_cut(**data_dict, phase="test")

    elif kwargs["dataset_names"][0] == "gale_mandarin":
        if prepare_dataset:
            create_gale_mandarin_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_gale_mandarin_cut(**data_dict, phase="train")
            dev_cuts = create_gale_mandarin_cut(**data_dict, phase="dev")
            test_cuts = create_gale_mandarin_cut(**data_dict, phase="test")

    elif kwargs["dataset_names"][0] == "santa_barbara":
        if prepare_dataset:
            create_santa_barbara_dataset(**data_dict)

        if get_cuts:
            test_cuts = create_santa_barbara_cut(**data_dict)

    elif kwargs["dataset_names"][0] == "callhome_english":
        if prepare_dataset:
            create_callhome_english_dataset(**data_dict)

        if get_cuts:
            train_cuts = create_callhome_english_cut(**data_dict, phase="train")
            dev_cuts = create_callhome_english_cut(**data_dict, phase="devtest")
            test_cuts = create_callhome_english_cut(**data_dict, phase="evaltest")

    elif kwargs["dataset_names"][0] == "gigaspeech":
        if prepare_dataset:
            create_gigaspeech_dataset(**data_dict)

        if get_cuts:
            # train_cuts = create_gigaspeech_cut(**data_dict, phase="train")
            # dev_cuts = create_gigaspeech_cut(**data_dict, phase="dev")
            test_cuts = create_gigaspeech_cut(**data_dict, phase="test")
