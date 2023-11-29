import os
from pathlib import Path
from typing import Dict, Optional, Union

import lhotse
from lhotse import load_manifest_lazy, CutSet, S3PRLSSLConfig, S3PRLSSL
from lhotse.recipes.dihard3 import prepare_dihard3


Pathlike = Union[Path, str]


def get_dihard3_cut(cut_set_path: Optional[Pathlike]) -> lhotse.CutSet:
    
    assert cut_set_path, f'cut_set_path must have a valid value. Currently, it is None'

    cut_set_path = Path(cut_set_path)
    assert cut_set_path.is_file(), f"No such file - cut_set_path: {cut_set_path}"

    return load_dihard3_cut(cut_set_path)

def load_dihard3_cut(cut_set_path: Pathlike) -> lhotse.CutSet:
    return load_manifest_lazy(cut_set_path)


def create_dihard3_cut(
    dev_audio_dir: Pathlike,
    eval_audio_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    feats_dir: Optional[Pathlike] = None,
    batch_duration: int = 600,
    phase: Optional[str] = 'dev',
    prepare_dataset: Optional[bool] = False,
    get_cuts: Optional[bool] = True
) -> lhotse.CutSet:

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/dihard3/manifests')
    feats_dir = Path(feats_dir) if feats_dir else Path('/export/c01/ashah108/vad/data/dihard3/feats')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    if prepare_dataset:
        prepare_dihard3(dev_audio_dir=dev_audio_dir, eval_audio_dir=eval_audio_dir, output_dir=output_dir)

    if not get_cuts:
        return None

    else:
        rec = load_manifest_lazy(f'{output_dir}/dihard3_recordings_{phase}.jsonl.gz')
        sup = load_manifest_lazy(f'{output_dir}/dihard3_supervisions_{phase}.jsonl.gz')
        base_filename = f'{phase}'

        cuts = CutSet.from_manifests(
            recordings=rec,
            supervisions=sup
        ).cut_into_windows(duration=5).filter(lambda cut: cut.duration > 3)

        cuts.to_file(f'{output_dir}/{base_filename}_cuts_ssl.jsonl.gz')

        # extractor = Fbank(FbankConfig(device='cuda'))

        # cuts = CutSet.from_file(f'{output_dir}/{base_filename}_cuts.jsonl.gz')
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

