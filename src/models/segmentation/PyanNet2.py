# MIT License
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from einops import rearrange

# from src.models.blocks.sincnet import SincNet
from src.utils.helper import pairwise, merge_dict


class PyanNet2(pl.LightningModule):
    """PyanNet2 segmentation model

    LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    # SINCNET_DEFAULTS = {"stride": 10}
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 4,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.5,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        lstm: dict = None,
        linear: dict = None,
        encoding_dim: int = 768,
        sample_rate: int = 16000,
        num_channels: int = 1,
    ):
        super(PyanNet2, self).__init__()

        # sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        # sincnet["sample_rate"] = sample_rate

        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True

        linear = merge_dict(self.LINEAR_DEFAULTS, linear)

        self.save_hyperparameters("lstm", "linear")

        # self.sincnet = SincNet(**self.hparams.sincnet)
        self.encoding_dim = encoding_dim
        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(encoding_dim, **multi_layer_lstm)

        else:
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        (
                            encoding_dim
                            if i == 0
                            else lstm["hidden_size"]
                            * (2 if lstm["bidirectional"] else 1)
                        ),
                        **one_layer_lstm
                    )
                    for i in range(num_layers)
                ]
            )

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        out_features: int = 1  # binary classification

        self.classifier = nn.Linear(in_features, out_features)
        self.activation = nn.Sigmoid()

    def forward(self, audio_feats: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        audio_feats : (batch, frames, features)

        Returns
        -------
        scores : (batch, frames, classes)
        """

        # outputs = self.sincnet(waveforms)  # (B, feature, frames)
        outputs = audio_feats

        if self.hparams.lstm["monolithic"]:
            outputs, _ = self.lstm(
                outputs
            )  # (B, frames, hidden_size) or (B, frames, 2 * hidden_size)

        else:
            # outputs = rearrange(outputs, "batch feature frames -> batch frames feature")
            for i, lstm in enumerate(self.lstm):
                outputs, _ = lstm(
                    outputs
                )  # (B, frames, hidden_size) or (B, frames, 2 * hidden_size)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    outputs = self.dropout(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))  # (B, frames, feats)

        return self.activation(self.classifier(outputs))  # (B, frames, 1)
