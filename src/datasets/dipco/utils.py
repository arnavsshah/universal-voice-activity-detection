import os
from pathlib import Path
from typing import Dict, Optional, Union

import lhotse
from lhotse import (
    load_manifest_lazy,
    CutSet,
    Fbank,
    FbankConfig,
    S3PRLSSLConfig,
    S3PRLSSL,
)
from lhotse.recipes import download_dipco, prepare_dipco

from src.utils.helper import get_cuts_subset


Pathlike = Union[Path, str]


def download_dipco_dataset(target_dir: Pathlike) -> None:
    target_dir = Path(target_dir) if target_dir else Path("/export/c01/ashah108")

    download_dipco(target_dir=target_dir)


def get_dipco_cut(cut_set_path: Optional[Pathlike]) -> lhotse.CutSet:

    assert cut_set_path, f"cut_set_path must have a valid value. Currently, it is None"

    cut_set_path = Path(cut_set_path)
    assert cut_set_path.is_file(), f"No such file - cut_set_path: {cut_set_path}"

    return load_dipco_cut(cut_set_path)


def load_dipco_cut(cut_set_path: Pathlike) -> lhotse.CutSet:
    return load_manifest_lazy(cut_set_path)


def create_dipco_dataset(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    mic: Optional[str] = "ihm",
    **kwargs,
) -> None:

    output_dir = (
        Path(output_dir)
        if output_dir
        else Path("/export/c01/ashah108/vad/data/dipco/manifests")
    )

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    prepare_dipco(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        mic=mic,
    )


def create_dipco_cut(
    output_dir: Optional[Pathlike] = None,
    mic: Optional[str] = "ihm",
    feats_dir: Optional[Pathlike] = None,
    batch_duration: int = 600,
    phase: Optional[str] = "dev",
    feature_extractor: Optional[str] = "wav2vec2",
    subset: Optional[bool] = False,
    subset_size: Optional[int] = 0,
    add_noise: Optional[bool] = False,
    noise_cut_set_path: Optional[Pathlike] = None,
    **kwargs,
) -> lhotse.CutSet:

    phases = ["dev", "eval"]
    assert phase in phases, f"Invalid phase: {phase}. Phase should be in {phases}"

    output_dir = (
        Path(output_dir)
        if output_dir
        else Path("/export/c01/ashah108/vad/data/dipco/manifests")
    )
    feats_dir = (
        Path(feats_dir)
        if feats_dir
        else Path("/export/c01/ashah108/vad/data/dipco/feats")
    )

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    rec = load_manifest_lazy(f"{output_dir}/dipco-{mic}_recordings_{phase}.jsonl.gz")
    sup = load_manifest_lazy(f"{output_dir}/dipco-{mic}_supervisions_{phase}.jsonl.gz")
    base_filename = f"{phase}_{mic}"

    if not os.path.exists(f"{output_dir}/{base_filename}_cuts_window.jsonl.gz"):
        cuts = CutSet.from_manifests(recordings=rec, supervisions=sup)

        # only 1 channel, but channel number is 1, thus it is treated as a multi-cut
        multi_cuts = cuts.multi_cuts
        mono_cuts = []
        for id, multi_cut in multi_cuts.items():
            mono_cuts.append(multi_cut.to_mono(mono_downmix=False)[0])

        cuts = CutSet.from_cuts(mono_cuts)

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

    if subset and phase == "dev":
        assert (
            subset_size > 0
        ), f"subset_size must be greater than 0. Currently, it is {subset_size}"
        cuts = get_cuts_subset(
            f"{output_dir}/{base_filename}_cuts_window.jsonl.gz", subset_size
        )
        to_append_filename += f"_subset_{subset_size}"

    if add_noise and phase == "dev":
        assert (
            noise_cut_set_path
        ), f"noise_cut_set_path must have a valid value. Currently, it is None"
        assert os.path.exists(
            noise_cut_set_path
        ), f"No such file - noise_cut_set_path: {noise_cut_set_path}"

        noise_cuts = load_manifest_lazy(noise_cut_set_path)
        cuts = cuts.mix(
            cuts=noise_cuts.to_eager(), snr=(10, 20), mix_prob=0.5, preserve_id="left"
        )
        to_append_filename += f"_with_noise"

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

    if phase == "dev":
        ids = []
        for cut in cuts:
            ids.append(cut.id)

        num_samples = len(ids)

        train_cuts = cuts.subset(cut_ids=ids[: int(0.8 * num_samples)])
        dev_cuts = cuts.subset(cut_ids=ids[int(0.8 * num_samples) :])

        train_cuts.to_file(
            f"{output_dir}/train_{mic}_cuts_{feature_extractor}{to_append_filename}.jsonl.gz"
        )
        dev_cuts.to_file(
            f"{output_dir}/dev_{mic}_cuts_{feature_extractor}{to_append_filename}.jsonl.gz"
        )

        return train_cuts, dev_cuts

    else:
        cuts.to_file(f"{output_dir}/eval_{mic}_cuts_{feature_extractor}.jsonl.gz")
        # cuts.describe()

        return cuts, None
