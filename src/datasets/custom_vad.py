from typing import Callable, Dict, Sequence

import numpy as np
import torch

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.dataset.collation import collate_vectors
from lhotse.utils import ifnone

from src.utils.receptive_field import get_num_frames


class CustomVadDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the voice activity detection task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'inputs': (B x T x F) tensor
            'input_lens': (B,) tensor
            'is_voice': (Conv_T x 1) tensor
            'cut': List[Cut]
        }
    """

    def __init__(
        self,
        input_strategy: BatchIO = PrecomputedFeatures(),
        cut_transforms: Sequence[Callable[[CutSet], CutSet]] = None,
        input_transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.input_strategy = input_strategy
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])

    def supervisions_feature_mask(self, cut: CutSet, num_samples: int) -> np.ndarray:
        # calculated before
        RECEPTIVE_FIELD_1, RECEPTIVE_FIELD_2 = 991, 1261
        STEP = RECEPTIVE_FIELD_2 - RECEPTIVE_FIELD_1  # 270
        HALF_DURATION = round(0.5 * RECEPTIVE_FIELD_1)  # 495

        # num_frames = 293
        num_frames = get_num_frames(num_samples)

        mask = np.zeros(
            num_frames,
            dtype=np.float32,
        )

        for supervision in cut.supervisions:

            # start_sample = (supervision.start - cut.start) * 16000
            # end_sample = (supervision.end - cut.start) * 16000

            start_sample = round(supervision.start * 16000)
            end_sample = round(supervision.end * 16000)

            st = (
                int((start_sample - HALF_DURATION) // STEP)
                if supervision.start > 0
                else 0
            )
            et = (
                int((end_sample - HALF_DURATION) // STEP)
                if supervision.end < cut.duration
                else num_frames
            )
            mask[st:et] = 1.0

        return mask

    def supervision_masks(self, cuts: CutSet, num_samples: int) -> torch.Tensor:
        return collate_vectors(
            [self.supervisions_feature_mask(cut, num_samples) for cut in cuts]
        )

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        validate(cuts)
        cuts = cuts.sort_by_duration()
        for tfnm in self.cut_transforms:
            cuts = tfnm(cuts)
        inputs, input_lens, filtered_cut_set = self.input_strategy(cuts)
        for tfnm in self.input_transforms:
            inputs = tfnm(inputs)
        is_voice = self.supervision_masks(filtered_cut_set, inputs.shape[1])

        return {
            "inputs": inputs,
            "input_lens": input_lens,
            "is_voice": is_voice,
            "cut": cuts,
        }
