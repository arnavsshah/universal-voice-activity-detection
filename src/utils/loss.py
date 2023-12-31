# MIT License
#
# Copyright (c) 2020-2021 CNRS
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
import torch.nn.functional as F


def interpolate(target: torch.Tensor, weight: torch.Tensor = None):
    """Interpolate weight to match target frame resolution

    Parameters
    ----------
    target : torch.Tensor
        Target with shape (batch_size, num_frames) or (batch_size, num_frames, num_classes)
    weight : torch.Tensor, optional
        Frame weight with shape (batch_size, num_frames_weight, 1).

    Returns
    -------
    weight : torch.Tensor
        Interpolated frame weight with shape (batch_size, num_frames, 1).
    """

    num_frames = target.shape[1]
    if weight is not None and weight.shape[1] != num_frames:
        weight = F.interpolate(
            weight.transpose(1, 2),
            size=num_frames,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return weight


def binary_cross_entropy(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None
) -> torch.Tensor:
    """Frame-weighted binary cross entropy

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction with shape (batch_size, num_frames, num_classes).
    target : torch.Tensor
        Target with shape (batch_size, num_frames) for binary or multi-class classification,
        or (batch_size, num_frames, num_classes) for multi-label classification.
    weight : (batch_size, num_frames, 1) torch.Tensor, optional
        Frame weight with shape (batch_size, num_frames, 1).

    Returns
    -------
    loss : torch.Tensor
    """

    # reshape target to (batch_size, num_frames, num_classes) even if num_classes is 1
    if len(target.shape) == 2:
        target = target.unsqueeze(dim=2)

    if weight is None:
        return F.binary_cross_entropy(prediction, target.float())

    else:
        # interpolate weight
        weight = interpolate(target, weight=weight)

        return F.binary_cross_entropy(
            prediction, target.float(), weight=weight.expand(target.shape)
        )
