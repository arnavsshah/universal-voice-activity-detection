from pathlib import Path
from typing import Dict, Optional, Union

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from lhotse import CutSet
from lhotse.dataset import VadDataset, SimpleCutSampler

from src.datasets import *

Pathlike = Union[Path, str]


class GlobalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_modules_params: Dict[str, list],
        max_duration: int,
        weights: Optional[Dict[str, float]] = None,
        stop_early: bool = True,
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
        weights : Optional[Dict[str, float]], optional
            Dictionary of weights for each dataset, by default None
        stop_early : bool, optional
            Whether to stop early when a batch exceeds the maximum duration, by default True
        """

        super().__init__()
        self.data_modules_params = data_modules_params
        self.max_duration = max_duration
        self.weights = list(weights.values()) if weights is not None else None
        self.stop_early = stop_early

        self.train_cuts_all, self.dev_cuts_all, self.test_cuts_all = [], [], []
    
    def prepare_data(self):

        for key, value in self.data_modules_params.items():
            if key == 'ami':
                self.train_cuts_all.append(get_ami_cut(**value[0]))
                self.dev_cuts_all.append(get_ami_cut(**value[1]))
                self.test_cuts_all.append(get_ami_cut(**value[2]))
            
            if key == 'dihard3':
                self.train_cuts_all.append(get_dihard3_cut(**value[0]))
                # self.dev_cuts_all.append(get_dihard3_cut(**value[0]))
                self.test_cuts_all.append(get_dihard3_cut(**value[1]))

            if key == 'switchboard':
                self.train_cuts_all.append(get_switchboard_cut(**value[0]))
                self.dev_cuts_all.append(get_switchboard_cut(**value[1]))
                self.test_cuts_all.append(get_switchboard_cut(**value[2]))

            if key == 'callhome_english':
                self.train_cuts_all.append(get_callhome_english_cut(**value[0]))
                self.dev_cuts_all.append(get_callhome_english_cut(**value[1]))
                self.test_cuts_all.append(get_callhome_english_cut(**value[2]))

            if key == 'callhome_egyptian':
                self.train_cuts_all.append(get_callhome_egyptian_cut(**value[0]))
                self.dev_cuts_all.append(get_callhome_egyptian_cut(**value[1]))
                self.test_cuts_all.append(get_callhome_egyptian_cut(**value[2]))

            if key == 'chime6':
                self.train_cuts_all.append(get_chime6_cut(**value[0]))
                self.dev_cuts_all.append(get_chime6_cut(**value[1]))
                self.test_cuts_all.append(get_chime6_cut(**value[2]))
            
            if key == 'tedlium':
                self.train_cuts_all.append(get_tedlium_cut(**value[0]))
                self.dev_cuts_all.append(get_tedlium_cut(**value[1]))
                self.test_cuts_all.append(get_tedlium_cut(**value[2]))

            if key == 'voxconverse':
                self.train_cuts_all.append(get_voxconverse_cut(**value[0]))
                self.dev_cuts_all.append(get_voxconverse_cut(**value[1]))
                self.test_cuts_all.append(get_voxconverse_cut(**value[2]))


        # combine all cuts
        if len(self.train_cuts_all) == 1:
            self.train_cuts = GlobalDataModule.combine_cuts(self.train_cuts_all)
            self.dev_cuts = GlobalDataModule.combine_cuts(self.dev_cuts_all)
            self.test_cuts = GlobalDataModule.combine_cuts(self.test_cuts_all)

        # multiplex cuts
        else:
            if self.weights:
                self.train_cuts = CutSet.mux(*self.train_cuts_all, weights=self.weights, stop_early=self.stop_early)
            else:
                self.train_cuts = CutSet.mux(*self.train_cuts_all, stop_early=self.stop_early)
            self.dev_cuts = CutSet.mux(*self.dev_cuts_all)
            self.test_cuts = CutSet.mux(*self.test_cuts_all)

    # def setup() ??

    def train_dataloader(self):
        dataset = VadDataset()
        sampler = SimpleCutSampler(self.train_cuts, max_duration=self.max_duration, shuffle=True)
        return DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=4)

    def val_dataloader(self):
        dataset = VadDataset()
        sampler = SimpleCutSampler(self.dev_cuts, max_duration=self.max_duration)
        return DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=4)

    def test_dataloader(self):
        dataset = VadDataset()
        sampler = SimpleCutSampler(self.test_cuts, max_duration=self.max_duration)
        return DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=4)

    @staticmethod
    def combine_cuts(cuts):
        if not cuts:
            return cuts

        combined_cuts = cuts[0]
        for cut in cuts[1:]:
            combined_cuts += cut
        return combined_cuts