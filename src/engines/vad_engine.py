from typing import Literal

import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

from src.models.segmentation.PyanNet import PyanNet
from src.utils.loss import binary_cross_entropy

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
    ):
        super(VadModel, self).__init__()
        
        self.model = PyanNet(**model_dict)
        self.model.build()

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()

        self.train_precision = BinaryPrecision()
        self.val_precision = BinaryPrecision()

        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()

        self.train_f1_score = BinaryF1Score()
        self.val_f1_score = BinaryF1Score()

    
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


    def training_step(self, batch, batch_idx):
        loss_dict, y_pred, y = self._common_step(batch, batch_idx, 'train')

        self.train_accuracy(y_pred, y)
        self.train_precision(y_pred, y)
        self.train_recall(y_pred, y)
        self.train_f1_score(y_pred, y)

        self.log_dict(
            {
                'train_loss': loss_dict['loss'],
                'train_acc_step': self.train_accuracy,
                'train_precision_step': self.train_precision,
                'train_recall_step': self.train_recall,
                'train_f1_score_step': self.train_f1_score,
            },
            batch_size=120, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            logger=True
        )

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        loss_dict, y_pred, y = self._common_step(batch, batch_idx, 'val')
        
        self.val_accuracy(y_pred, y)
        self.val_precision(y_pred, y)
        self.val_recall(y_pred, y)
        self.val_f1_score(y_pred, y)

        self.log_dict(
            {
                'val_loss': loss_dict['loss'],
                'val_acc_step': self.val_accuracy,
                'val_precision_step': self.val_precision,
                'val_recall_step': self.val_recall,
                'val_f1_score_step': self.val_f1_score,
            },
            batch_size=120, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            logger=True
        )

        return loss_dict['loss']


    def on_train_epoch_end(self):
        self.log_dict({
            'train_acc_epoch': self.train_accuracy.compute(),
            'train_precision_epoch': self.train_precision.compute(),
            'train_recall_epoch': self.train_recall.compute(),
            'train_f1_score_epoch': self.train_f1_score.compute(),
        })

    def on_val_epoch_end(self):
        self.log_dict({
            'val_acc_epoch': self.val_accuracy.compute(),
            'val_precision_epoch': self.val_precision.compute(),
            'val_recall_epoch': self.val_recall.compute(),
            'val_f1_score_epoch': self.val_f1_score.compute(),
        })


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
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

