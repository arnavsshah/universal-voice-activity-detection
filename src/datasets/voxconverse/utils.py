import os
from pathlib import Path
from typing import Dict, Optional, Union

import lhotse
from lhotse import load_manifest_lazy, CutSet, S3PRLSSLConfig, S3PRLSSL, FbankConfig, Fbank
from lhotse.recipes import prepare_voxconverse


Pathlike = Union[Path, str]


def get_voxconverse_cut(cut_set_path: Optional[Pathlike]) -> lhotse.CutSet:
    
    assert cut_set_path, f'cut_set_path must have a valid value. Currently, it is None'

    cut_set_path = Path(cut_set_path)
    assert cut_set_path.is_file(), f"No such file - cut_set_path: {cut_set_path}"

    return load_voxconverse_cut(cut_set_path)

def load_voxconverse_cut(cut_set_path: Pathlike) -> lhotse.CutSet:
    return load_manifest_lazy(cut_set_path)


def create_voxconverse_dataset(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    **kwargs
) -> None:

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/voxconverse/manifests')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    prepare_voxconverse(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        split_test=True
    )


def create_voxconverse_cut(
    output_dir: Optional[Pathlike] = None,
    feats_dir: Optional[Pathlike] = None,
    batch_duration: int = 600,
    phase: Optional[str] = 'train',
    **kwargs
) -> lhotse.CutSet:

    phases = ["train", "dev", "test"]
    assert phase in phases, f'Invalid phase: {phase}. Phase should be in {phases}'

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/voxconverse/manifests')
    feats_dir = Path(feats_dir) if feats_dir else Path('/export/c01/ashah108/vad/data/voxconverse/feats')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    rec = load_manifest_lazy(f'{output_dir}/voxconverse_recordings_{phase}.jsonl.gz')
    sup = load_manifest_lazy(f'{output_dir}/voxconverse_supervisions_{phase}.jsonl.gz')
    base_filename = f'{phase}'

    cuts = CutSet.from_manifests(
        recordings=rec,
        supervisions=sup
    ).cut_into_windows(duration=5).filter(lambda cut: cut.duration > 3)

    cuts.to_file(f'{output_dir}/{base_filename}_cuts_ssl.jsonl.gz')

    # extractor = Fbank(FbankConfig(
    #     sampling_rate=16000,
    #     device='cuda'
    # ))
    extractor = S3PRLSSL(S3PRLSSLConfig(device='cuda', ssl_model='wav2vec2'))

    cuts = cuts.compute_and_store_features_batch(
        extractor=extractor,
        storage_path=f'{feats_dir}/{base_filename}_feats_ssl',
        batch_duration=batch_duration,
        num_workers=4,
        overwrite=True
    ).pad(duration=5.0)


    cuts.to_file(f'{output_dir}/{base_filename}_cuts_feats_ssl.jsonl.gz')
    # cuts.describe()

    return cuts

