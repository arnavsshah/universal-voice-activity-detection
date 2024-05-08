import os
from pathlib import Path
from copy import deepcopy
import json
import math
from pprint import pprint

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lhotse import load_manifest_lazy, CutSet

from src.engines.vad_engine import VadModel
from src.datasets.data_module import GlobalDataModule
from src.utils.helper import prepare_data_module_params, median_filter
from src.utils.receptive_field import get_num_frames

from config.config import *


def predict_vad_sincnet(**kwargs):
    """
    Predict with the VAD model

    Parameters
    ----------
     **kwargs : dict
        Keyword arguments for configuring the dataset, model and training. The following keys are supported:

        - 'model_name' (str): Name of the model to be used.
        - 'dataset_name' (str): Name of the dataset to operate on.
        - 'supported_models' (list): A list of supported model names.
        - 'supported_datasets' (list): A list of supported dataset names.
        - 'seed' (int): seed to ensure reproducibility
        - 'experiments_dir' (str): Directory path for storing experiment-related files and model checkpoints.
        - 'predict_dataset_names' - (list) list of datasets to be used for prediction (for now, one only).
            Each dataset has as separate dictionary of model-specific configuration.
            Some of them are listed below. Others might have dataset-specific paramteres
            - 'train_cut_set_path' (str): Path to the training cut set file.
            - 'dev_cut_set_path' (str): Path to the development cut set file.
            - 'test_cut_set_path' (str): Path to the test cut set file.
        - 'max_duration' (float): Maximum duration of each batch of audio data in the dataloader in seconds.
        - 'model_dict' (dict): Dictionary of model-specific configuration.
        - 'device' (str): Device to use for processing (e.g., 'cpu' or 'gpu').
        - 'max_epochs' (int): Maximum number of training epochs for the model.
        - 'is_wandb' (bool): If True, data is logged to weights and biases, else locally
        - 'wandb' (dict): Dictionary of wandb-logging-specific configuration
            - 'project' (str): project name to log to
            - 'run' (str): current wandb run name
    """

    # torch.set_printoptions(profile="full")

    assert (
        kwargs["model_name"] in kwargs["supported_models"]
    ), f"Invalid model {kwargs['model_name']}. Model should be one of {kwargs['supported_models']}"

    for dataset_name in kwargs["predict_dataset_names"]:
        assert (
            dataset_name in kwargs["supported_datasets"]
        ), f"Invalid dataset {dataset_name}. Dataset should be one of {kwargs['supported_datasets']}"

    assert kwargs["load_checkpoint"], "Please provide a checkpoint path to load from"

    pl.seed_everything(kwargs["seed"], workers=True)

    if not os.path.exists(kwargs["experiments_dir"]):
        Path(kwargs["experiments_dir"]).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(kwargs["predict_output_dir"]):
        Path(kwargs["predict_output_dir"]).mkdir(parents=True, exist_ok=True)

    logger = None
    frame_shift = kwargs["frame_shift"]

    model = VadModel.load_from_checkpoint(checkpoint_path=kwargs["checkpoint_path"])

    trainer = pl.Trainer(
        accelerator=kwargs["device"],
        devices=1,
        default_root_dir=kwargs["experiments_dir"],
        logger=logger,
        deterministic=True,
    )

    for dataset_name in kwargs["predict_dataset_names"]:
        data_modules_params = prepare_data_module_params([dataset_name], kwargs)
        data_module = GlobalDataModule(
            data_modules_params,
            kwargs["max_duration"],
            custom_vad=kwargs["feature_extractor"] == "sincnet",
        )
        data_module.setup()

        # train_preds = trainer.predict(model, data_module.train_dataloader())  # list of (batch_size, num_frames, 1)
        # dev_preds = trainer.predict(model, data_module.val_dataloader())  # list of (batch_size, num_frames, 1)
        test_preds = trainer.predict(
            model, data_module.test_dataloader()
        )  # list of (batch_size, num_frames, 1)

        # train_preds = torch.cat(train_preds, axis=0)  # (batch, num_frames, 1)
        # dev_preds = torch.cat(dev_preds, axis=0)  # (batch, num_frames, 1)
        test_preds = torch.cat(test_preds, axis=0)  # (batch, num_frames, 1)

        # torch.save(train_preds, os.path.join(kwargs['predict_output_dir'], f'train_preds_{dataset_name}.pt'))
        # torch.save(dev_preds, os.path.join(kwargs['predict_output_dir'], f'dev_preds_{dataset_name}.pt'))
        torch.save(
            test_preds,
            os.path.join(
                kwargs["predict_output_dir"], f"sincnet_test_preds_{dataset_name}.pt"
            ),
        )

        if dataset_name == "ami":
            buffer = 0
            # split, split_display = True, 'split'
            split, split_display = False, "no_split"
            test_cuts = get_new_cuts(
                dataset_name=dataset_name,
                phase="test",
                tensor_file_name=f"sincnet_test_preds_{dataset_name}.pt",
                recordings_path="/export/c01/ashah108/vad/data/ami/manifests/ami-ihm_recordings_test.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/ami/manifests/test_ihm_cuts_og.jsonl.gz",
                predict_output_dir=kwargs["predict_output_dir"],
                output_filename=f"cuts_test_ihm_{dataset_name}_buffer-{buffer}_{split_display}.jsonl.gz",
                buffer=buffer,
                split=split,
            )

        elif dataset_name == "tedlium":
            buffer = 0
            # split, split_display = True, "split"
            split, split_display = False, "no_split"
            test_cuts = get_new_cuts(
                dataset_name=dataset_name,
                phase="test",
                tensor_file_name=f"sincnet_test_preds_{dataset_name}.pt",
                recordings_path="/export/c01/ashah108/vad/data/tedlium/manifests/tedlium_recordings_test.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/tedlium/manifests/test_cuts_og.jsonl.gz",
                predict_output_dir=kwargs["predict_output_dir"],
                output_filename=f"sincnet_test_cuts_{dataset_name}_buffer-{buffer}_{split_display}.jsonl.gz",
                buffer=buffer,
                split=split,
                alignment_path="/export/c01/ashah108/vad/force_align/supervisions_obj_with_alignment_test.json",
            )

        elif dataset_name == "switchboard":
            buffer = 0
            # split, split_display = True, 'split'
            split, split_display = False, "no_split"
            test_cuts = get_new_cuts(
                dataset_name=dataset_name,
                phase="test",
                tensor_file_name=f"sincnet_test_preds_{dataset_name}.pt",
                recordings_path="/export/c01/ashah108/vad/data/eval2000/manifests/eval2000_recordings_all.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/eval2000/manifests/test_cuts_og.jsonl.gz",
                predict_output_dir=kwargs["predict_output_dir"],
                output_filename=f"sincnet_test_cuts_{dataset_name}_buffer-{buffer}_{split_display}.jsonl.gz",
                buffer=buffer,
                split=split,
            )

        elif dataset_name == "voxconverse":
            buffer = 0
            # split, split_display = True, 'split'
            split, split_display = False, "no_split"
            test_cuts = get_new_cuts(
                dataset_name=dataset_name,
                phase="test",
                tensor_file_name=f"sincnet_test_preds_{dataset_name}.pt",
                recordings_path="/export/c01/ashah108/vad/data/voxconverse/manifests/voxconverse_recordings_test.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/voxconverse/manifests/test_cuts_og.jsonl.gz",
                predict_output_dir=kwargs["predict_output_dir"],
                output_filename=f"sincnet_test_cuts_{dataset_name}_buffer-{buffer}_{split_display}.jsonl.gz",
                buffer=buffer,
                split=split,
            )

        elif dataset_name == "chime6":
            buffer = 0
            # split, split_display = True, 'split'
            split, split_display = False, "no_split"
            test_cuts = get_new_cuts(
                dataset_name=dataset_name,
                phase="test",
                tensor_file_name=f"sincnet_test_preds_{dataset_name}.pt",
                recordings_path="/export/c01/ashah108/vad/data/chime6/manifests/chime6-mdm_recordings_eval.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/chime6/manifests/eval_mdm_cuts_og.jsonl.gz",
                predict_output_dir=kwargs["predict_output_dir"],
                output_filename=f"sincnet_test_cuts_{dataset_name}_buffer-{buffer}_{split_display}.jsonl.gz",
                buffer=buffer,
                split=split,
            )

        elif dataset_name == "dihard3":
            buffer = 0
            # split, split_display = True, 'split'
            split, split_display = False, "no_split"
            test_cuts = get_new_cuts(
                dataset_name=dataset_name,
                phase="test",
                tensor_file_name=f"sincnet_test_preds_{dataset_name}.pt",
                recordings_path="/export/c01/ashah108/vad/data/dihard3/manifests/dihard3_recordings_eval.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/dihard3/manifests/eval_cuts_og.jsonl.gz",
                predict_output_dir=kwargs["predict_output_dir"],
                output_filename=f"sincnet_test_cuts_{dataset_name}_buffer-{buffer}_{split_display}.jsonl.gz",
                buffer=buffer,
                split=split,
            )

        elif dataset_name == "dipco":
            buffer = 0
            # split, split_display = True, 'split'
            split, split_display = False, "no_split"
            test_cuts = get_new_cuts(
                dataset_name=dataset_name,
                phase="test",
                tensor_file_name=f"sincnet_test_preds_{dataset_name}.pt",
                recordings_path="/export/c01/ashah108/vad/data/dipco/manifests/dipco-ihm_recordings_eval.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/dipco/manifests/eval_ihm_cuts_og.jsonl.gz",
                predict_output_dir=kwargs["predict_output_dir"],
                output_filename=f"sincnet_test_cuts_{dataset_name}_buffer-{buffer}_{split_display}.jsonl.gz",
                buffer=buffer,
                split=split,
            )

        elif dataset_name == "voxceleb":
            buffer = 0
            # split, split_display = True, 'split'
            split, split_display = False, "no_split"
            test_cuts = get_new_cuts(
                dataset_name=dataset_name,
                phase="test",
                tensor_file_name=f"sincnet_test_preds_{dataset_name}.pt",
                recordings_path="/export/c01/ashah108/vad/data/voxceleb/manifests/voxceleb_recordings_test.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/voxceleb/manifests/test_cuts_og.jsonl.gz",
                predict_output_dir=kwargs["predict_output_dir"],
                output_filename=f"sincnet_test_cuts_{dataset_name}_buffer-{buffer}_{split_display}.jsonl.gz",
                buffer=buffer,
                split=split,
            )

        elif dataset_name == "gale_arabic":
            buffer = 0
            # split, split_display = True, 'split'
            split, split_display = False, "no_split"
            test_cuts = get_new_cuts(
                dataset_name=dataset_name,
                phase="test",
                tensor_file_name=f"sincnet_test_preds_{dataset_name}.pt",
                recordings_path="/export/c01/ashah108/vad/data/gale_arabic/manifests/gale-arabic_recordings_test.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/gale_arabic/manifests/test_cuts_og.jsonl.gz",
                predict_output_dir=kwargs["predict_output_dir"],
                output_filename=f"sincnet_test_cuts_{dataset_name}_buffer-{buffer}_{split_display}.jsonl.gz",
                buffer=buffer,
                split=split,
            )

        elif dataset_name == "gale_mandarin":
            buffer = 0
            # split, split_display = True, 'split'
            split, split_display = False, "no_split"
            test_cuts = get_new_cuts(
                dataset_name=dataset_name,
                phase="test",
                tensor_file_name=f"sincnet_test_preds_{dataset_name}.pt",
                recordings_path="/export/c01/ashah108/vad/data/gale_mandarin/manifests/gale-mandarin_recordings_dev.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/gale_mandarin/manifests/test_cuts_og.jsonl.gz",
                predict_output_dir=kwargs["predict_output_dir"],
                output_filename=f"sincnet_test_cuts_{dataset_name}_buffer-{buffer}_{split_display}.jsonl.gz",
                buffer=buffer,
                split=split,
            )

        elif dataset_name == "santa_barbara":
            buffer = 0
            # split, split_display = True, 'split'
            split, split_display = False, "no_split"
            test_cuts = get_new_cuts(
                dataset_name=dataset_name,
                phase="test",
                tensor_file_name=f"sincnet_test_preds_{dataset_name}.pt",
                recordings_path="/export/c01/ashah108/vad/data/santa_barbara/manifests/sbcsae_rec.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/santa_barbara/manifests/santa_barbara_cuts_og.jsonl.gz",
                predict_output_dir=kwargs["predict_output_dir"],
                output_filename=f"sincnet_test_cuts_{dataset_name}_buffer-{buffer}_{split_display}.jsonl.gz",
                buffer=buffer,
                split=split,
            )


def get_new_cuts(
    dataset_name,
    phase,
    tensor_file_name,
    recordings_path,
    cuts_path,
    predict_output_dir,
    output_filename,
    buffer=0,
    split=False,
    alignment_path=None,
):

    if alignment_path:
        supervisions_obj = json.load(open(alignment_path, "r"))

    preds = torch.load(
        os.path.join(predict_output_dir, tensor_file_name)
    )  # (batch, num_frames, 1)
    preds_flat = preds.reshape(-1)  # (batch * num_frames)

    recordings = [obj.to_dict() for obj in load_manifest_lazy(recordings_path)]
    all_cuts = [obj for obj in load_manifest_lazy(cuts_path)]

    sup_dict, sup_set = {}, set()
    empty_cut, in_sup, not_in_sup, exceed_sup, in_multiple_sup = 0, 0, 0, 0, 0
    det_err_avg, false_alarm_avg, miss_det_avg = 0, 0, 0

    for cut in all_cuts:
        for sup in cut.supervisions:
            sup_dict[sup.id] = [sup.start, sup.duration, sup.text, 0]

    recording_tensor = [
        {"recording_id": obj["id"], "duration": obj["duration"]} for obj in recordings
    ]

    start, end = 0, 0
    for i, obj in enumerate(recording_tensor):
        start = end
        end = start + math.ceil(get_num_frames(16000 * obj["duration"])) + 1
        end = min(end, len(preds_flat))

        recording_tensor[i]["tensor"] = preds_flat[start:end]

    new_cuts = []

    for i, obj in enumerate(recording_tensor):
        cut = all_cuts[i]

        gt_intervals, pred_intervals = [], []

        for sup in cut.supervisions:
            gt_intervals.append((sup.start, sup.start + sup.duration))

        start = None
        for k, value in enumerate(obj["tensor"]):
            if value >= 0.5:  # threshold 0.5
                if start is None:
                    start = k
            else:
                if start is not None:
                    end = k - 1

                    start_time, end_time = get_timestamp_from_sample_boundary(
                        start, end, obj["duration"]
                    )
                    if end_time - start_time > 0.0:
                        pred_intervals.append((start_time, end_time))
                    start = None

        if start is not None:
            end = len(obj["tensor"]) - 1
            start_time, end_time = get_timestamp_from_sample_boundary(
                start, end, obj["duration"]
            )
            if end_time - start_time > 0.0:
                pred_intervals.append((start_time, end_time))

        merged_intervals = merge_intervals_with_buffer(
            pred_intervals, recording_tensor[i]["duration"], buffer
        )

        window_intervals = merged_intervals
        if split:
            window_intervals = split_into_windows(merged_intervals, window=10)

        gt_tensor = get_binary_tensor(gt_intervals, obj["duration"])
        pred_tensor = get_binary_tensor(window_intervals, obj["duration"])

        false_alarm_temp = get_false_alarm(gt_tensor, pred_tensor)
        miss_det_temp = get_missed_detection(gt_tensor, pred_tensor)
        det_err_temp = false_alarm_temp + miss_det_temp

        false_alarm_avg += false_alarm_temp
        miss_det_avg += miss_det_temp
        det_err_avg += det_err_temp

        supervisions_index = cut.index_supervisions(index_mixed_tracks=False)

        for j in range(len(window_intervals)):
            abs_start, abs_end = window_intervals[j]
            new_cut = cut.truncate(
                offset=abs_start,
                duration=abs_end - abs_start,
                keep_excessive_supervisions=True,
                _supervisions_index=supervisions_index,
            ).with_id(f"{cut.id}-{j}")

            if len(new_cut.supervisions) != 0:

                for sup in new_cut.supervisions:
                    if sup.id in sup_dict:
                        in_sup += 1
                        sup_set.add(sup.id)
                        sup_dict[sup.id][3] += 1

                        cut_start, sup_start = round(new_cut.start, 2), round(
                            sup_dict[sup.id][0], 2
                        )
                        cut_end, sup_end = round(
                            new_cut.start + new_cut.duration, 2
                        ), round(sup_dict[sup.id][0] + sup_dict[sup.id][1], 2)

                        if cut_start > sup_start or cut_end < sup_end:
                            exceed_sup += 1
                        if sup_dict[sup.id][3] > 1:
                            in_multiple_sup += 1

                start, duration = 0, new_cut.duration

                new_cut.supervisions[0].start = start
                new_cut.supervisions[0].duration = duration

                text = ""
                for sup in new_cut.supervisions:
                    if sup.text is not None:
                        text += sup.text + " "
                text = text.strip()

                text_alignment = ""
                if alignment_path:
                    for sup in supervisions_obj[obj["recording_id"]]["supervisions"]:
                        if is_overlap(
                            sup["start"],
                            sup["start"] + sup["duration"],
                            abs_start,
                            abs_end,
                        ):
                            for alignment in sup["alignment"]:
                                if (
                                    sup["start"] + alignment["start"] >= abs_start
                                    and sup["start"] + alignment["end"] <= abs_end
                                ):
                                    text_alignment += alignment["word"] + " "
                        elif sup["start"] > abs_end:
                            break
                    text = text_alignment.strip()

                new_cut.supervisions[0].text = text

                # for k in range(1, len(new_cut.supervisions)):
                #     if new_cut.supervisions[k].text is not None:
                #         new_cut.supervisions[0].text += " " + new_cut.supervisions[k].text
                #     else:
                #         new_cut.supervisions[0].text = ""

                new_cut.supervisions = [new_cut.supervisions[0]]
                new_cuts.append(new_cut)

            else:
                empty_cut += 1

    cuts = CutSet.from_cuts(new_cuts)
    cuts.to_file(os.path.join(predict_output_dir, output_filename))

    false_alarm_avg = false_alarm_avg / len(recordings)
    miss_det_avg = miss_det_avg / len(recordings)
    det_err_avg = det_err_avg / len(recordings)

    not_in_sup = len(sup_dict.keys()) - len(sup_set)

    print(f"Dataset: {dataset_name}, Buffer: {buffer}, Phase: {phase}")
    print("\n")
    print(f"Detection Error Rate: {det_err_avg}")
    print(f"False Alarm Rate: {false_alarm_avg}")
    print(f"Missed Detection Rate: {miss_det_avg}")
    print("----------------")
    print(f"Total Supervisions: {len(sup_dict.keys())}")
    print(f"Supervisions in new cuts: {in_sup}")
    print(f"Unique Supervisions in new cuts: {len(sup_set)}")
    print(f"Supervisions not in new cuts: {not_in_sup}")
    print(f"Supervisions exceeding new cuts: {exceed_sup}")
    print(f"Supervisions in multiple new cuts: {in_multiple_sup}")
    print(f"Empty cuts: {empty_cut}")
    print("\n")
    print("\n")


def get_timestamp_from_sample_boundary(start, end, duration):
    # calculated before
    RECEPTIVE_FIELD_1, RECEPTIVE_FIELD_2 = 991, 1261
    STEP = RECEPTIVE_FIELD_2 - RECEPTIVE_FIELD_1  # 270
    HALF_DURATION = round(0.5 * RECEPTIVE_FIELD_1)  # 495

    start_time = round((start * STEP + HALF_DURATION) / 16000)
    end_time = round((end * STEP + HALF_DURATION) / 16000)

    start_time = max(start_time, 0)
    end_time = min(end_time, duration)

    return start_time, end_time


def merge_intervals_with_buffer(intervals, total_duration, buffer):
    if len(intervals) == 0:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])

    intervals_with_buffer = []
    for interval in intervals:
        intervals_with_buffer.append(
            [max(interval[0] - buffer, 0), min(interval[1] + buffer, total_duration)]
        )

    merged_intervals = []
    start, end = intervals_with_buffer[0]
    for i in range(1, len(intervals_with_buffer)):
        if intervals_with_buffer[i][0] <= end:
            end = intervals_with_buffer[i][1]
        else:
            merged_intervals.append([start, end])
            start, end = intervals_with_buffer[i]
    merged_intervals.append([start, end])
    return merged_intervals


# split only if greater than window size
def split_into_windows(intervals, window=10):
    new_intervals = []
    for interval in intervals:
        start, end = interval
        while end - start > window:
            new_intervals.append([start, start + window])
            start += window
        if end - start > 0.1:
            new_intervals.append([start, end])
    return new_intervals


def is_overlap(start1, end1, start2, end2):
    return end1 >= start2 and end2 >= start1


def get_binary_tensor(intervals, total_duration):
    # # calculated before
    # RECEPTIVE_FIELD_1, RECEPTIVE_FIELD_2 = 991, 1261
    # STEP = RECEPTIVE_FIELD_2 - RECEPTIVE_FIELD_1  # 270
    # HALF_DURATION = round(0.5 * RECEPTIVE_FIELD_1)  # 495

    # num_frames = get_num_frames(total_duration * 16000)
    # tensor = torch.zeros(num_frames)

    # for interval in intervals:
    #     start, end = interval
    #     start_sample = round(start * 16000)
    #     end_sample = round(end * 16000)

    #     st = int((start_sample - HALF_DURATION) // STEP) if start > 0 else 0
    #     et = (
    #         int((end_sample - HALF_DURATION) // STEP)
    #         if end < total_duration
    #         else num_frames
    #     )

    #     tensor[st:et] = 1

    # return tensor

    frame_shift = 0.02
    tensor = torch.zeros(math.ceil(total_duration / frame_shift))

    for interval in intervals:
        start, end = interval
        start, end = int(start / frame_shift), int(end / frame_shift)

        tensor[start:end] = 1

    return tensor


def get_false_alarm(gt_tensor, pred_tensor):
    false_alarm = torch.sum(torch.logical_and(gt_tensor == 0, pred_tensor == 1))
    return false_alarm / len(gt_tensor)


def get_missed_detection(gt_tensor, pred_tensor):
    miss_det = torch.sum(torch.logical_and(gt_tensor == 1, pred_tensor == 0))
    return miss_det / len(gt_tensor)
