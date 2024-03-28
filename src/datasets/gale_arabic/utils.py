import os
from pathlib import Path
from typing import Dict, Optional, Union

import lhotse
from lhotse import load_manifest_lazy, CutSet, Fbank, FbankConfig, S3PRLSSLConfig, S3PRLSSL
from lhotse.recipes import prepare_gale_arabic


Pathlike = Union[Path, str]


def get_gale_arabic_cut(cut_set_path: Optional[Pathlike]) -> lhotse.CutSet:
    
    assert cut_set_path, f'cut_set_path must have a valid value. Currently, it is None'

    cut_set_path = Path(cut_set_path)
    assert cut_set_path.is_file(), f"No such file - cut_set_path: {cut_set_path}"

    return load_gale_arabic_cut(cut_set_path)


def load_gale_arabic_cut(cut_set_path: Pathlike) -> lhotse.CutSet:
    return load_manifest_lazy(cut_set_path)


def create_gale_arabic_dataset(
    audio_dirs: Pathlike,
    transcript_dirs: Pathlike,
    output_dir: Optional[Pathlike] = None,
    **kwargs
) -> None:
    
    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/gale_arabic/manifests')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    prepare_gale_arabic(audio_dirs=audio_dirs, transcript_dirs=transcript_dirs, output_dir=output_dir)


def create_gale_arabic_cut(
    output_dir: Optional[Pathlike] = None,
    feats_dir: Optional[Pathlike] = None,
    batch_duration: int = 600,
    phase: Optional[str] = 'train',
    feature_extractor: Optional[str] = 'wav2vec2',
    **kwargs
) -> lhotse.CutSet:

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/gale_arabic/manifests')
    feats_dir = Path(feats_dir) if feats_dir else Path('/export/c01/ashah108/vad/data/gale_arabic/feats')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    rec = load_manifest_lazy(f'{output_dir}/gale-arabic_recordings_{phase}.jsonl.gz')
    sup = load_manifest_lazy(f'{output_dir}/gale-arabic_supervisions_{phase}.jsonl.gz')
    base_filename = f'{phase}'

    cuts = CutSet.from_manifests(
        recordings=rec,
        supervisions=sup
    )
    
    cuts.to_file(f'{output_dir}/{base_filename}_cuts_og.jsonl.gz')

    cuts_trim = cuts.trim_to_supervisions(keep_overlapping=False, keep_all_channels=False)
    cuts_trim.to_file(f'{output_dir}/{base_filename}_cuts_trim.jsonl.gz')

    cuts = cuts.cut_into_windows(duration=5).filter(lambda cut: cut.duration > 3)

    cuts.to_file(f'{output_dir}/{base_filename}_cuts_window.jsonl.gz')

    cuts = cuts.subset(first=len(cuts.to_eager()))  # progress bar doesn't appear otherwise
    
    if feature_extractor == 'fbank':
        extractor = Fbank(FbankConfig(sampling_rate=16000, device='cuda'))
    else:
        extractor = S3PRLSSL(S3PRLSSLConfig(device='cuda', ssl_model=feature_extractor))

    cuts = cuts.compute_and_store_features_batch(
        extractor=extractor,
        storage_path=f'{feats_dir}/{base_filename}_{feature_extractor}',
        batch_duration=batch_duration,
        num_workers=4,
        overwrite=True
    ).pad(duration=5.0)


    if phase == "train":
        ids = []
        for cut in cuts:
            ids.append(cut.id)

        num_samples = len(ids)

        train_cuts = cuts.subset(cut_ids=ids[:int(0.8 * num_samples)])
        dev_cuts = cuts.subset(cut_ids=ids[int(0.8 * num_samples):])
        
        train_cuts.to_file(f'{output_dir}/train_cuts_{feature_extractor}.jsonl.gz')
        dev_cuts.to_file(f'{output_dir}/dev_cuts_{feature_extractor}.jsonl.gz')

        return train_cuts, dev_cuts
    
    else:
        cuts.to_file(f'{output_dir}/test_cuts_{feature_extractor}.jsonl.gz')
        # cuts.describe()

        return cuts, None

    return cuts

