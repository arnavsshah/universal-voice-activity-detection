from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from lhotse import CutSet, Fbank
from lhotse.dataset import VadDataset, SimpleCutSampler, CutMix

from src.datasets.custom_vad import CustomVadDataset
from lhotse.dataset.input_strategies import AudioSamples

from src.datasets import *
from src.utils.receptive_field import get_num_frames

Pathlike = Union[Path, str]


class GlobalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_modules_params: Dict[str, list],
        max_duration: int,
        weights_dict: Optional[Dict[str, float]] = None,
        stop_early: bool = True,
        custom_vad: bool = True,
    ):
        """
        Generate a global data module to combine all the given datasets
        Parameters
        ----------
        data_modules_params : Dict[str, list]
            Dictionary containing data module parameters for all datasets
            Each dictionary key is a dataset name and each value is a list of parameters which includes:
                - train_data_dict : Dict[str, str]
                - dev_data_dict : Dict[str, str]
                - test_data_dict : Dict[str, str]
        max_duration : int
            Maximum duration of a batch in seconds
        weights_dict : Optional[Dict[str, float]], optional
            Dictionary of weights for each dataset, by default None
        stop_early : bool, optional
            Whether to stop early when a batch exceeds the maximum duration, by default True
        custom_vad : bool, optional
            Whether to use custom VAD dataset, by default True
        """

        super().__init__()
        # self.prepare_data_per_node = True

        self.data_modules_params = data_modules_params
        self.max_duration = max_duration
        self.stop_early = stop_early
        self.custom_vad = custom_vad

        self.train_cuts_all, self.dev_cuts_all, self.test_cuts_all = [], [], []

        self.weights_dict = weights_dict
        self.weights = [] if weights_dict is not None else None

    # def prepare_data(self):
    # pass

    def setup(self, stage: str = None):
        for key, value in self.data_modules_params.items():

            if self.weights is not None:
                self.weights.append(self.weights_dict[key])

            if key == "ami":
                self.train_cuts_all.append(get_ami_cut(**value[0]))
                self.dev_cuts_all.append(get_ami_cut(**value[1]))
                self.test_cuts_all.append(get_ami_cut(**value[2]))

            if key == "tedlium":
                self.train_cuts_all.append(get_tedlium_cut(**value[0]))
                self.dev_cuts_all.append(get_tedlium_cut(**value[1]))
                self.test_cuts_all.append(get_tedlium_cut(**value[2]))

            if key == "switchboard":
                self.train_cuts_all.append(get_switchboard_cut(**value[0]))
                self.dev_cuts_all.append(get_switchboard_cut(**value[1]))
                self.test_cuts_all.append(get_switchboard_cut(**value[2]))

            if key == "voxconverse":
                self.train_cuts_all.append(get_voxconverse_cut(**value[0]))
                self.dev_cuts_all.append(get_voxconverse_cut(**value[1]))
                self.test_cuts_all.append(get_voxconverse_cut(**value[2]))

            if key == "chime6":
                self.train_cuts_all.append(get_chime6_cut(**value[0]))
                self.dev_cuts_all.append(get_chime6_cut(**value[1]))
                self.test_cuts_all.append(get_chime6_cut(**value[2]))

            if key == "dihard3":
                self.train_cuts_all.append(get_dihard3_cut(**value[0]))
                # self.dev_cuts_all.append(get_dihard3_cut(**value[0]))
                self.test_cuts_all.append(get_dihard3_cut(**value[1]))

            if key == "dipco":
                # self.train_cuts_all.append(get_dipco_cut(**value[0]))
                # self.dev_cuts_all.append(get_dipco_cut(**value[1]))
                self.test_cuts_all.append(get_dipco_cut(**value[0]))

            if key == "gale_arabic":
                self.train_cuts_all.append(get_gale_arabic_cut(**value[0]))
                self.dev_cuts_all.append(get_gale_arabic_cut(**value[1]))
                self.test_cuts_all.append(get_gale_arabic_cut(**value[2]))

            if key == "gale_mandarin":
                self.train_cuts_all.append(get_gale_mandarin_cut(**value[0]))
                self.dev_cuts_all.append(get_gale_mandarin_cut(**value[1]))
                self.test_cuts_all.append(get_gale_mandarin_cut(**value[2]))

            if key == "santa_barbara":
                self.test_cuts_all.append(get_santa_barbara_cut(**value[0]))

            if key == "callhome_english":
                self.train_cuts_all.append(get_callhome_english_cut(**value[0]))
                self.dev_cuts_all.append(get_callhome_english_cut(**value[1]))
                self.test_cuts_all.append(get_callhome_english_cut(**value[2]))

            if key == "gigaspeech":
                # self.train_cuts_all.append(get_gigaspeech_cut(**value[0]))
                # self.dev_cuts_all.append(get_gigaspeech_cut(**value[1]))
                self.test_cuts_all.append(get_gigaspeech_cut(**value[2]))

        # combine all cuts
        if len(self.train_cuts_all) == 1:
            self.train_cuts = self.combine_cuts(self.train_cuts_all)
            self.dev_cuts = self.combine_cuts(self.dev_cuts_all)
            self.test_cuts = self.combine_cuts(self.test_cuts_all)

        elif len(self.test_cuts_all) == 1:
            self.test_cuts = self.combine_cuts(self.test_cuts_all)

        # multiplex cuts
        else:
            if self.weights != None:
                self.train_cuts = CutSet.mux(
                    *self.train_cuts_all,
                    weights=self.weights,
                    stop_early=self.stop_early,
                )
            else:
                self.train_cuts = CutSet.mux(
                    *self.train_cuts_all, stop_early=self.stop_early
                )

            self.dev_cuts = CutSet.mux(*self.dev_cuts_all)
            self.test_cuts = CutSet.mux(*self.test_cuts_all)

    # def teardown(self, stage: Optional[str] = None):
    # pass

    def train_dataloader(self):
        dataset = (
            CustomVadDataset(input_strategy=AudioSamples(fault_tolerant=True))
            if self.custom_vad
            else VadDataset()
        )
        sampler = SimpleCutSampler(
            self.train_cuts,
            max_duration=self.max_duration,
            shuffle=True,
            drop_last=True,
        )
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=4,
        )

    def val_dataloader(self):
        dataset = (
            CustomVadDataset(input_strategy=AudioSamples(fault_tolerant=True))
            if self.custom_vad
            else VadDataset()
        )
        sampler = SimpleCutSampler(
            self.dev_cuts, max_duration=self.max_duration, drop_last=True
        )
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=4,
        )

    def test_dataloader(self):
        dataset = (
            CustomVadDataset(input_strategy=AudioSamples(fault_tolerant=True))
            if self.custom_vad
            else VadDataset()
        )
        sampler = SimpleCutSampler(self.test_cuts, max_duration=self.max_duration)
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=4,
        )

    def combine_cuts(self, cuts):
        if not cuts:
            return cuts

        combined_cuts = cuts[0]
        for cut in cuts[1:]:
            combined_cuts += cut
        return combined_cuts
