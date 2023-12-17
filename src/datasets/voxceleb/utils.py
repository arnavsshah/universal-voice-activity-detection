import os
from pathlib import Path
from typing import Dict, Optional, Union

import lhotse
from lhotse import load_manifest_lazy, CutSet, S3PRLSSLConfig, S3PRLSSL
from lhotse.recipes import prepare_voxceleb


Pathlike = Union[Path, str]


def get_voxceleb_cut(cut_set_path: Optional[Pathlike]) -> lhotse.CutSet:
    
    assert cut_set_path, f'cut_set_path must have a valid value. Currently, it is None'

    cut_set_path = Path(cut_set_path)
    assert cut_set_path.is_file(), f"No such file - cut_set_path: {cut_set_path}"

    return load_voxceleb_cut(cut_set_path)

def load_voxceleb_cut(cut_set_path: Pathlike) -> lhotse.CutSet:
    return load_manifest_lazy(cut_set_path)


def create_voxceleb_dataset(
    voxceleb1_root: Pathlike,
    voxceleb2_root: Pathlike,
    output_dir: Optional[Pathlike] = None,
    **kwargs
) -> lhotse.CutSet:

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/voxceleb/manifests')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    prepare_voxceleb(
        voxceleb1_root=voxceleb1_root,
        voxceleb2_root=voxceleb2_root,
        output_dir=output_dir,
        num_jobs=6,
    )


def create_voxceleb_cut(
    output_dir: Optional[Pathlike] = None,
    feats_dir: Optional[Pathlike] = None,
    batch_duration: int = 600,
    phase: Optional[str] = 'train',
    **kwargs
) -> lhotse.CutSet:

    phases = ["train", "dev", "test"]
    assert phase in phases, f'Invalid phase: {phase}. Phase should be in {phases}'

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/voxceleb/manifests')
    feats_dir = Path(feats_dir) if feats_dir else Path('/export/c01/ashah108/vad/data/voxceleb/feats')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    rec = load_manifest_lazy(f'{output_dir}/voxceleb_recordings_{phase}.jsonl.gz')
    sup = load_manifest_lazy(f'{output_dir}/voxceleb_supervisions_{phase}.jsonl.gz')
    base_filename = f'{phase}'

    cuts = CutSet.from_manifests(
        recordings=rec,
        supervisions=sup
    )
    
    multi_cuts = cuts.multi_cuts
    mono_cuts = []
    for id, multi_cut in multi_cuts.items():
        mono_cuts.append(multi_cut.to_mono(mono_downmix=False)[0])
        
    cuts = CutSet.from_cuts(mono_cuts).cut_into_windows(duration=5).filter(lambda cut: cut.duration > 3)

    cuts.to_file(f'{output_dir}/{base_filename}_cuts_ssl.jsonl.gz')

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

