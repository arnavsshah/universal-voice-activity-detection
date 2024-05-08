import os
from pathlib import Path
from typing import Dict, Optional, Union, Sequence

import lhotse
from lhotse import (
    load_manifest_lazy,
    CutSet,
    Fbank,
    FbankConfig,
    S3PRLSSLConfig,
    S3PRLSSL,
)
from lhotse.recipes import prepare_gigaspeech

from src.utils.helper import get_cuts_subset


Pathlike = Union[Path, str]


def get_gigaspeech_cut(cut_set_path: Optional[Pathlike]) -> lhotse.CutSet:

    assert cut_set_path, f"cut_set_path must have a valid value. Currently, it is None"

    cut_set_path = Path(cut_set_path)
    assert cut_set_path.is_file(), f"No such file - cut_set_path: {cut_set_path}"

    return load_gigaspeech_cut(cut_set_path)


def load_gigaspeech_cut(cut_set_path: Pathlike) -> lhotse.CutSet:
    return load_manifest_lazy(cut_set_path)


def create_gigaspeech_dataset(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dataset_parts: Union[str, Sequence[str]] = ("S", "DEV", "TEST"),
    **kwargs,
) -> None:

    output_dir = (
        Path(output_dir)
        if output_dir
        else Path("/export/c01/ashah108/vad/data/gigaspeech/manifests")
    )

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    prepare_gigaspeech(
        corpus_dir=corpus_dir, output_dir=output_dir, dataset_parts=dataset_parts
    )


def create_gigaspeech_cut(
    output_dir: Optional[Pathlike] = None,
    feats_dir: Optional[Pathlike] = None,
    batch_duration: int = 600,
    phase: Optional[str] = "train",
    feature_extractor: Optional[str] = "wav2vec2",
    subset: Optional[bool] = False,
    subset_size: Optional[int] = 0,
    add_noise: Optional[bool] = False,
    noise_cut_set_path: Optional[Pathlike] = None,
    **kwargs,
) -> lhotse.CutSet:

    assert phase in [
        "train",
        "dev",
        "test",
    ], f"Invalid phase: {phase}. Must be one of ['train', 'dev', 'test']"

    output_dir = (
        Path(output_dir)
        if output_dir
        else Path("/export/c01/ashah108/vad/data/gigaspeech/manifests")
    )
    feats_dir = (
        Path(feats_dir)
        if feats_dir
        else Path("/export/c01/ashah108/vad/data/gigaspeech/feats")
    )

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    filename_phase = "S" if phase == "train" else phase.upper()
    base_filename = f"{phase}"

    if not os.path.exists(f"{output_dir}/{base_filename}_cuts_window.jsonl.gz"):
        # rec = load_manifest_lazy(f"{output_dir}/gigaspeech_recordings_{phase}.jsonl.gz")
        # sup = load_manifest_lazy(f"{output_dir}/gigaspeech_supervisions_{phase}.jsonl.gz")
        cuts = load_manifest_lazy(
            f"{output_dir}/gigaspeech_cuts_{filename_phase}.jsonl.gz"
        )

        cuts.to_file(f"{output_dir}/{base_filename}_cuts_og.jsonl.gz")

        cuts_trim = cuts.trim_to_supervisions(
            keep_overlapping=False, keep_all_channels=False
        )
        cuts_trim.to_file(f"{output_dir}/{base_filename}_cuts_trim.jsonl.gz")

        cuts = cuts.cut_into_windows(duration=5).filter(lambda cut: cut.duration > 3)

        cuts.to_file(f"{output_dir}/{base_filename}_cuts_window.jsonl.gz")

    else:
        cuts = load_manifest_lazy(f"{output_dir}/{base_filename}_cuts_window.jsonl.gz")

    to_append_filename = ""

    if phase == "train":
        if subset:
            assert (
                subset_size > 0
            ), f"subset_size must be greater than 0. Currently, it is {subset_size}"
            cuts = get_cuts_subset(
                f"{output_dir}/{base_filename}_cuts_window.jsonl.gz", subset_size
            )
            to_append_filename += f"_subset_{subset_size}"

        if add_noise:
            assert (
                noise_cut_set_path
            ), f"noise_cut_set_path must have a valid value. Currently, it is None"
            assert os.path.exists(
                noise_cut_set_path
            ), f"No such file - noise_cut_set_path: {noise_cut_set_path}"

            noise_cuts = load_manifest_lazy(noise_cut_set_path)
            cuts = cuts.mix(
                cuts=noise_cuts.to_eager(),
                snr=(10, 20),
                mix_prob=0.5,
                preserve_id="left",
            )
            to_append_filename += f"_with_noise"

        if subset or add_noise:
            cuts.to_file(
                f"{output_dir}/{base_filename}_cuts_window{to_append_filename}.jsonl.gz"
            )

    cuts = cuts.subset(
        first=len(cuts.to_eager())
    )  # progress bar doesn't appear otherwise

    if feature_extractor == "fbank":
        extractor = Fbank(FbankConfig(sampling_rate=16000, device="cuda"))
    else:
        extractor = S3PRLSSL(S3PRLSSLConfig(device="cuda", ssl_model=feature_extractor))

    cuts = cuts.compute_and_store_features_batch(
        extractor=extractor,
        storage_path=f"{feats_dir}/{base_filename}_{feature_extractor}{to_append_filename}",
        batch_duration=batch_duration,
        num_workers=4,
        overwrite=True,
    ).pad(duration=5.0)

    cuts.to_file(
        f"{output_dir}/{base_filename}_cuts_{feature_extractor}{to_append_filename}.jsonl.gz"
    )
    # cuts.describe()

    return cuts
