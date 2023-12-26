from typing import Literal

import torch
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryStatScores,
)

import wandb

from src.models.segmentation.PyanNet import PyanNet
from src.utils.loss import binary_cross_entropy
from src.utils.helper import median_filter

from config.config import *


class VadModel(pl.LightningModule):
    """
    Universal voice activity detection (VAD) model

    Parameters
    ----------
    model_name: (str) Name of the model to be used
    model_dict : (dict) Dictionary of model-specific configuration.
    """

    def __init__(
        self,
        model_name: str = 'PyanNet',
        model_dict: dict = {},
        learning_rate: float = 1e-3,
    ):
        super(VadModel, self).__init__()

        self.model = PyanNet(**model_dict)
        self.model.build()

        self.learning_rate = learning_rate

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

        self.train_precision = BinaryPrecision()
        self.val_precision = BinaryPrecision()
        self.test_precision = BinaryPrecision()

        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()
        self.test_recall = BinaryRecall()

        self.train_f1_score = BinaryF1Score()
        self.val_f1_score = BinaryF1Score()
        self.test_f1_score = BinaryF1Score()

        self.train_stat_scores = BinaryStatScores()
        self.val_stat_scores = BinaryStatScores()
        self.test_stat_scores = BinaryStatScores()

        self.total_train_duration = 0
        self.total_val_duration = 0

    
    def forward(self, audio_feats: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        audio_feats : (batch, frames, features)

        Returns
        -------
        scores : (batch, frames, classes)
        """

        return self.model(audio_feats)


    def predict_step(self, batch, batch_idx , dataloader_idx=None):
        return self.model(batch['inputs'])
        

    def training_step(self, batch, batch_idx):
        loss_dict, y_pred, y = self._common_step(batch, batch_idx, 'train')

        # with torch.no_grad():
        #     x = torch.where(y_pred < 0.5, 0, 1).to('cuda')
        #     print(x.sum().item())

        self.train_accuracy(y_pred.squeeze(-1), y)
        self.train_precision(y_pred.squeeze(-1), y)
        self.train_recall(y_pred.squeeze(-1), y)
        self.train_f1_score(y_pred.squeeze(-1), y)
        stat_scores = self.train_stat_scores(y_pred.squeeze(-1), y)
        denominator = y_pred.shape[0] * y_pred.shape[1]
        self.total_train_duration += denominator

        self.log_dict(
            {
                'train_detection_error_rate': (stat_scores[1] + stat_scores[3]) / denominator,
                'train_false_alarm': stat_scores[1] / denominator,
                'train_missed_detection': stat_scores[3] / denominator,
                'train_acc': self.train_accuracy,
                'train_precision': self.train_precision,
                'train_recall': self.train_recall,
                'train_f1_score': self.train_f1_score,
                'train_denominator': denominator,
                'train_loss': loss_dict['loss'],
            },
            batch_size=120, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=False, 
            logger=True
        )

        # do not backpropagate, do not update gradients. COnvert y_pred with threshold is 0.5 to either 0 or 1. Then, sum y_pred and y to print it out.
        # with torch.no_grad():
        #     x = torch.where(y_pred < 0.5, 0, 1).to('cuda')
        #     print(x.sum().item(), y.sum().item())

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        loss_dict, y_pred, y = self._common_step(batch, batch_idx, 'val')
        
        self.val_accuracy(y_pred.squeeze(-1), y)
        self.val_precision(y_pred.squeeze(-1), y)
        self.val_recall(y_pred.squeeze(-1), y)
        self.val_f1_score(y_pred.squeeze(-1), y)
        stat_scores = self.val_stat_scores(y_pred.squeeze(-1), y)
        denominator = y_pred.shape[0] * y_pred.shape[1]
        self.total_val_duration += denominator

        self.log_dict(
            {
                'val_detection_error_rate': (stat_scores[1] + stat_scores[3]) / denominator,
                'val_false_alarm': stat_scores[1] / denominator,
                'val_missed_detection': stat_scores[3] / denominator,
                'val_acc': self.val_accuracy,
                'val_precision': self.val_precision,
                'val_recall': self.val_recall,
                'val_f1_score': self.val_f1_score,
                'val_denominator': float(denominator),
                'val_loss': loss_dict['loss'],
            },
            batch_size=120, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=False, 
            logger=True
        )

        # # do not backpropagate, do not update gradients. COnvert y_pred with threshold is 0.5 to either 0 or 1. Then, sum y_pred and y to print it out.
        # with torch.no_grad():
        #     x = torch.where(y_pred < 0.5, 0, 1).to('cuda')
        #     print(x.sum().item(), y.sum().item())

        return loss_dict['loss']

    
    def test_step(self, batch, batch_idx):
        loss_dict, y_pred, y = self._common_step(batch, batch_idx, 'val')

        y_pred = median_filter(y_pred.squeeze(-1))  # (batch, frames)
        y_pred = y_pred.unsqueeze(-1)  # (batch, frames, 1)
        
        self.test_accuracy(y_pred.squeeze(-1), y)
        self.test_precision(y_pred.squeeze(-1), y)
        self.test_recall(y_pred.squeeze(-1), y)
        self.test_f1_score(y_pred.squeeze(-1), y)
        stat_scores = self.test_stat_scores(y_pred.squeeze(-1), y)
        denominator = y_pred.shape[0] * y_pred.shape[1]

        self.log_dict(
            {
                'test_detection_error_rate': (stat_scores[1] + stat_scores[3]) / denominator,
                'test_false_alarm': stat_scores[1] / denominator,
                'test_missed_detection': stat_scores[3] / denominator,
                'test_acc': self.test_accuracy,
                'test_precision': self.test_precision,
                'test_recall': self.test_recall,
                'test_f1_score': self.test_f1_score,
                'test_loss': loss_dict['loss'],
                'test_den': denominator,
            },
            batch_size=120, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=True, 
            logger=True
        )

        return loss_dict['loss']


    def on_train_epoch_end(self):
        stat_scores = self.train_stat_scores.compute()
        # denominator = y_pred.shape[0] * y_pred.shape[1]
        
        self.log_dict({
            'train_detection_error_rate': (stat_scores[1] + stat_scores[3]) / self.total_train_duration,
            'train_false_alarm': stat_scores[1] / self.total_train_duration,
            'train_missed_detection': stat_scores[3] / self.total_train_duration,
            'train_acc': self.train_accuracy.compute(),
            'train_precision': self.train_precision.compute(),
            'train_recall': self.train_recall.compute(),
            'train_f1_score': self.train_f1_score.compute(),
            'train_denominator': self.total_train_duration,
        })

        self.total_train_duration = 0

    def on_val_epoch_end(self):
        stat_scores = self.val_stat_scores.compute()
        # denominator = y_pred.shape[0] * y_pred.shape[1]

        self.log_dict({
            'val_detection_error_rate': (stat_scores[1] + stat_scores[3]) / self.total_val_duration,
            'val_false_alarm': stat_scores[1] / self.total_val_duration,
            'val_missed_detection': stat_scores[3] / self.total_val_duration,
            'val_acc': self.val_accuracy.compute(),
            'val_precision': self.val_precision.compute(),
            'val_recall': self.val_recall.compute(),
            'val_f1_score': self.val_f1_score.compute(),
            'val_denominator': self.total_val_duration,
        })

        self.total_val_duration = 0


    def _common_step(self, batch, batch_idx, stage: Literal['train', 'val']):
        """
        Default step to be executed in training or validation loop

        Parameters
        ----------
        batch : (usually) dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        stage : {"train", "val"}
            "train" for training step, "val" for validation step

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss}
        """

        assert stage in ['train', 'val'], f"stage can only be in {['train', 'val']}. You have stage={stage}"

        # forward pass
        y_pred = self.model(batch['inputs'])
        y = batch['is_voice']

        # compute loss
        loss = binary_cross_entropy(y_pred, y, weight=None)

        if torch.isnan(loss):
            return None

        return {'loss': loss}, y_pred, y


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

