import faulthandler

faulthandler.enable()
import torch

from src.scripts import *
from data import *

from config.config import load_config


def main(config):

    if config.task == "wer":
        calc_wer()

    elif config.task == "download":
        download_data(**config)

    elif config.task == "prepare":
        prepare_data(**config)

    elif config.task == "compute_feats":
        prepare_feats(**config)

    elif config.task == "cuts_subset":
        prepare_cuts_subset(**config)

    elif config.task == "test_data":
        test_data(**config)

    elif config.task == "run":
        if config.function == "train":
            train_vad(**config)
        elif config.function == "test":
            test_vad(**config)
        elif config.function == "predict":
            predict_vad(**config)
        elif config.function == "predict_sincnet":
            predict_vad_sincnet(**config)
        elif config.function == "predict_dihard3":
            predict_dihard3_split(**config)
        elif config.function == "other_vad":
            other_vad_metrics(**config)


if __name__ == "__main__":

    print("GPU:", torch.cuda.is_available())

    # a = torch.ones(1).to("cuda")  # to avoid race condition, use cuda device at the start
    # print(a)

    config = load_config()
    main(config)
