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

from config.config import *


def other_vad_metrics(**kwargs):
    """
    Get metrics for other VAD
    """

    # torch.set_printoptions(profile="full")

    for dataset_name in kwargs["predict_dataset_names"]:
        assert (
            dataset_name in kwargs["supported_datasets"]
        ), f"Invalid dataset {dataset_name}. Dataset should be one of {kwargs['supported_datasets']}"

    frame_shift = kwargs["frame_shift"]

    for dataset_name in kwargs["predict_dataset_names"]:

        if dataset_name == "ami":
            test_cuts = get_metrics(
                dataset_name=dataset_name,
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/ami/manifests/ami-ihm_recordings_test.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/ami/manifests/test_ihm_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )

        elif dataset_name == "tedlium":
            test_cuts = get_metrics(
                dataset_name=dataset_name,
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/tedlium/manifests/tedlium_recordings_test.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/tedlium/manifests/test_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )

        elif dataset_name == "switchboard":
            test_cuts = get_metrics(
                dataset_name="eval2000",
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/eval2000/manifests/eval2000_recordings_all.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/eval2000/manifests/test_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )

        elif dataset_name == "voxconverse":
            test_cuts = get_metrics(
                dataset_name=dataset_name,
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/voxconverse/manifests/voxconverse_recordings_test.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/voxconverse/manifests/test_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )

        elif dataset_name == "chime6":
            test_cuts = get_metrics(
                dataset_name=dataset_name,
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/chime6/manifests/chime6-mdm_recordings_eval.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/chime6/manifests/eval_mdm_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )

        elif dataset_name == "dihard3":
            test_cuts = get_metrics(
                dataset_name=dataset_name,
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/dihard3/manifests/dihard3_recordings_eval.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/dihard3/manifests/eval_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )

        elif dataset_name == "dipco":
            test_cuts = get_metrics(
                dataset_name=dataset_name,
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/dipco/manifests/dipco-ihm_recordings_eval.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/dipco/manifests/eval_ihm_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )

        elif dataset_name == "voxceleb":
            test_cuts = get_metrics(
                dataset_name=dataset_name,
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/voxceleb/manifests/voxceleb_recordings_test.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/voxceleb/manifests/test_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )

        elif dataset_name == "gale_arabic":
            test_cuts = get_metrics(
                dataset_name=dataset_name,
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/gale_arabic/manifests/gale-arabic_recordings_test.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/gale_arabic/manifests/test_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )

        elif dataset_name == "gale_mandarin":
            test_cuts = get_metrics(
                dataset_name=dataset_name,
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/gale_mandarin/manifests/gale-mandarin_recordings_dev.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/gale_mandarin/manifests/test_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )

        elif dataset_name == "santa_barbara":
            test_cuts = get_metrics(
                dataset_name=dataset_name,
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/santa_barbara/manifests/sbcsae_rec.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/santa_barbara/manifests/santa_barbara_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )

        elif dataset_name == "callhome_english":
            test_cuts = get_metrics(
                dataset_name=dataset_name,
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/callhome_english/manifests/callhome-english_recordings_evaltest.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/callhome_english/manifests/evaltest_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )

        elif dataset_name == "gigaspeech":
            test_cuts = get_metrics(
                dataset_name=dataset_name,
                phase="test",
                recordings_path="/export/c01/ashah108/vad/data/gigaspeech/manifests/gigaspeech_recordings_TEST.jsonl.gz",
                cuts_path="/export/c01/ashah108/vad/data/gigaspeech/manifests/test_cuts_og.jsonl.gz",
                frame_shift=frame_shift,
            )


def get_metrics(
    dataset_name,
    phase,
    recordings_path,
    cuts_path,
    labels_path="/export/c01/ashah108/baseline_vad/pyannote_output",
    frame_shift=0.02,
):

    recordings = load_manifest_lazy(recordings_path)
    cuts_full = [obj for obj in load_manifest_lazy(cuts_path)]

    sup_dict, sup_set = {}, set()
    empty_cut, in_sup, not_in_sup, exceed_sup, in_multiple_sup = 0, 0, 0, 0, 0

    for cut in cuts_full:
        for sup in cut.supervisions:
            sup_dict[sup.id] = [sup.start, sup.duration, sup.text, 0]

    det_err_avg, false_alarm_avg, miss_det_avg = 0, 0, 0

    for i, rec in enumerate(recordings):
        cut = cuts_full[i]
        rec_id = rec.id

        gt_intervals, intervals = [], []

        for sup in cut.supervisions:
            gt_intervals.append((sup.start, sup.start + sup.duration))

        with open(f"{labels_path}/{dataset_name}/{rec_id}.txt", "r") as f:
            lines = f.readlines()

            for line in lines:
                _, start, end, label = line.strip().split()

                if label == "SPEECH":
                    start, end = float(start), float(end)

                    if start < 0:
                        start = 0
                    if end > rec.duration:
                        end = rec.duration

                    intervals.append((start, end))

        merged_intervals = merge_intervals(intervals, rec.duration)

        gt_tensor = get_binary_tensor(gt_intervals, rec.duration, frame_shift)
        pred_tensor = get_binary_tensor(intervals, rec.duration, frame_shift)

        false_alarm_temp = get_false_alarm(gt_tensor, pred_tensor)
        miss_det_temp = get_missed_detection(gt_tensor, pred_tensor)
        det_err_temp = false_alarm_temp + miss_det_temp

        false_alarm_avg += false_alarm_temp
        miss_det_avg += miss_det_temp
        det_err_avg += det_err_temp

        supervisions_index = cut.index_supervisions(index_mixed_tracks=False)

        for j in range(len(merged_intervals)):
            abs_start, abs_end = merged_intervals[j]
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

                        # print(cut_start, cut_end, sup_start, sup_end)

                start, duration = 0, new_cut.duration

            else:
                empty_cut += 1

    false_alarm_avg = false_alarm_avg / len(recordings)
    miss_det_avg = miss_det_avg / len(recordings)
    det_err_avg = det_err_avg / len(recordings)

    not_in_sup = len(sup_dict.keys()) - len(sup_set)

    print(f"Dataset: {dataset_name}, Phase: {phase}")
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


def merge_intervals(intervals, total_duration):
    if len(intervals) == 0:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])

    corrected_intervals = []
    for interval in intervals:
        corrected_intervals.append(
            [max(interval[0], 0), min(interval[1], total_duration)]
        )

    merged_intervals = []
    start, end = corrected_intervals[0]
    for i in range(1, len(corrected_intervals)):
        if corrected_intervals[i][0] <= end:
            end = corrected_intervals[i][1]
        else:
            merged_intervals.append([start, end])
            start, end = corrected_intervals[i]
    merged_intervals.append([start, end])
    return merged_intervals


def get_binary_tensor(intervals, total_duration, frame_shift):
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
