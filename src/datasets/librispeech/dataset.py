from pathlib import Path
from typing import Dict, Optional, Union

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from lhotse.dataset import VadDataset, SimpleCutSampler

from src.datasets.librispeech.utils import get_librispeech_cut

Pathlike = Union[Path, str]


class LibrispeechDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dict: dict,
        dev_data_dict: dict,
        test_data_dict: dict,
        max_duration: int            
    ):
        super().__init__()
        self.train_data_dict = train_data_dict
        self.dev_data_dict = dev_data_dict
        self.test_data_dict = test_data_dict
        self.max_duration = max_duration
    
    def prepare_data(self):
        # self.train_cuts = get_librispeech_cut(**self.train_data_dict, phase='train', prepare_dataset=False)
        self.dev_cuts = get_librispeech_cut(**self.dev_data_dict, phase='dev', prepare_dataset=False)
        self.test_cuts = get_librispeech_cut(**self.test_data_dict, phase='test', prepare_dataset=False)

    # def setup() ??

    def train_dataloader(self):
        dataset = VadDataset()
        sampler = SimpleCutSampler(self.dev_cuts, max_duration=self.max_duration)
        return DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=10)

    def val_dataloader(self):
        dataset = VadDataset()
        sampler = SimpleCutSampler(self.dev_cuts, max_duration=self.max_duration)
        return DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=4)

    def test_dataloader(self):
        dataset = VadDataset()
        sampler = SimpleCutSampler(self.test_cuts, max_duration=self.max_duration)
        return DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=4)

