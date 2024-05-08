import os
from pathlib import Path
from typing import Dict, Optional, Union, List

import lhotse
from lhotse import (
    load_manifest_lazy,
    CutSet,
    Fbank,
    FbankConfig,
    S3PRLSSLConfig,
    S3PRLSSL,
)
from lhotse.manipulation import combine
from lhotse.recipes import prepare_single_babel_language

from src.utils.helper import get_cuts_subset


Pathlike = Union[Path, str]


def get_babel_cut(cut_set_path: Optional[Pathlike]) -> lhotse.CutSet:

    assert cut_set_path, f"cut_set_path must have a valid value. Currently, it is None"

    cut_set_path = Path(cut_set_path)
    assert cut_set_path.is_file(), f"No such file - cut_set_path: {cut_set_path}"

    return load_babel_cut(cut_set_path)


def load_babel_cut(cut_set_path: Pathlike) -> lhotse.CutSet:
    return load_manifest_lazy(cut_set_path)


def create_babel_dataset(
    corpus_dir: Pathlike,
    relative_lang_paths: List[Pathlike],
    output_dir: Optional[Pathlike] = None,
    **kwargs,
) -> None:

    output_dir = (
        Path(output_dir)
        if output_dir
        else Path("/export/c01/ashah108/vad/data/babel/manifests")
    )

    output_lang_dir = output_dir / "languages"

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(output_lang_dir):
        output_lang_dir.mkdir(parents=True, exist_ok=True)

    # splits = ["train", "dev", "eval"]
    splits = ["train", "dev"]
    recs = {split: [] for split in splits}
    sups = {split: [] for split in splits}

    langs = [
        relative_lang_path.split("/")[0].split("-")[1].title()
        for relative_lang_path in relative_lang_paths
    ]

    for lang_path in relative_lang_paths:
        corpus_lang_dir = corpus_dir / Path(lang_path)
        assert (
            corpus_lang_dir.is_dir()
        ), f"No such directory - lang_path: {corpus_lang_dir}"

        manifests = prepare_single_babel_language(
            corpus_dir=corpus_lang_dir, output_dir=output_lang_dir
        )

    for split in splits:
        for lang in langs:
            rec = load_manifest_lazy(
                f"{output_lang_dir}/babel-{lang}_recordings_{split}.jsonl.gz"
            )
            sup = load_manifest_lazy(
                f"{output_lang_dir}/babel-{lang}_supervisions_{split}.jsonl.gz"
            )
            recs[split].append(rec)
            sups[split].append(sup)

    for split in splits:
        rec_to_save = combine(recs[split])
        sups_to_save = combine(sups[split])

        rec_to_save.to_file(f"{output_dir}/babel_recordings_{split}.jsonl.gz")
        sups_to_save.to_file(f"{output_dir}/babel_supervisions_{split}.jsonl.gz")


def create_babel_cut(
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
        "eval",
    ], f"Invalid phase: {phase}. Must be one of ['train', 'dev', 'eval']"

    output_dir = (
        Path(output_dir)
        if output_dir
        else Path("/export/c01/ashah108/vad/data/babel/manifests")
    )
    feats_dir = (
        Path(feats_dir)
        if feats_dir
        else Path("/export/c01/ashah108/vad/data/babel/feats")
    )

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    rec = load_manifest_lazy(f"{output_dir}/babel_recordings_{phase}.jsonl.gz")
    sup = load_manifest_lazy(f"{output_dir}/babel_supervisions_{phase}.jsonl.gz")
    base_filename = f"{phase}"

    cuts = CutSet.from_manifests(recordings=rec, supervisions=sup)

    cuts = cuts.resample(16000)

    cuts.to_file(f"{output_dir}/{base_filename}_cuts_og.jsonl.gz")

    cuts_trim = cuts.trim_to_supervisions(
        keep_overlapping=False, keep_all_channels=False
    )
    cuts_trim.to_file(f"{output_dir}/{base_filename}_cuts_trim.jsonl.gz")

    cuts = cuts.cut_into_windows(duration=5).filter(lambda cut: cut.duration > 3)

    cuts.to_file(f"{output_dir}/{base_filename}_cuts_window.jsonl.gz")

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
