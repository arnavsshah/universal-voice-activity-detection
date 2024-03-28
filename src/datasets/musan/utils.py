import os
from pathlib import Path
from typing import Dict, Optional, Union, Sequence

import lhotse
from lhotse import load_manifest_lazy, CutSet, Fbank, FbankConfig, S3PRLSSLConfig, S3PRLSSL
from lhotse.recipes import prepare_musan


Pathlike = Union[Path, str]


def get_musan_cut(cut_set_path: Optional[Pathlike]) -> lhotse.CutSet:
    
    assert cut_set_path, f'cut_set_path must have a valid value. Currently, it is None'

    cut_set_path = Path(cut_set_path)
    assert cut_set_path.is_file(), f"No such file - cut_set_path: {cut_set_path}"

    return load_musan_cut(cut_set_path)

def load_musan_cut(cut_set_path: Pathlike) -> lhotse.CutSet:
    return load_manifest_lazy(cut_set_path)


def create_musan_dataset(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    parts: Sequence[str] = ("music", "speech", "noise"),
    **kwargs
) -> None:
    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/musan/manifests')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    prepare_musan(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        parts=parts,
    )


def create_musan_cut(
    output_dir: Optional[Pathlike] = None,
    feats_dir: Optional[Pathlike] = None,
    batch_duration: int = 600,
    feature_extractor: Optional[str] = 'wav2vec2',
    **kwargs
) -> lhotse.CutSet:

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/musan/manifests')
    feats_dir = Path(feats_dir) if feats_dir else Path('/export/c01/ashah108/vad/data/musan/feats')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    rec_music = load_manifest_lazy(f'{output_dir}/musan_recordings_music.jsonl.gz')
    rec_noise = load_manifest_lazy(f'{output_dir}/musan_recordings_noise.jsonl.gz')
    rec_speech = load_manifest_lazy(f'{output_dir}/musan_recordings_speech.jsonl.gz')

    cuts = CutSet.from_manifests(
        recordings=(rec_music + rec_noise + rec_speech)
    )
    
    cuts.to_file(f'{output_dir}/musan_cuts_og.jsonl.gz')

    cuts = cuts.cut_into_windows(duration=5).filter(lambda cut: cut.duration > 3)
    
    cuts.to_file(f'{output_dir}/musan_cuts_window.jsonl.gz')

    cuts = cuts.subset(first=len(cuts.to_eager()))  # progress bar doesn't appear otherwise
    
    if feature_extractor == 'fbank':
        extractor = Fbank(FbankConfig(sampling_rate=16000, device='cuda'))
    else:
        extractor = S3PRLSSL(S3PRLSSLConfig(device='cuda', ssl_model=feature_extractor))

    cuts = cuts.compute_and_store_features_batch(
        extractor=extractor,
        storage_path=f'{feats_dir}/musan_{feature_extractor}',
        batch_duration=batch_duration,
        num_workers=4,
        overwrite=True
    ).pad(duration=5.0)


    cuts.to_file(f'{output_dir}/musan_cuts_{feature_extractor}.jsonl.gz')
    # cuts.describe()

    return cuts

