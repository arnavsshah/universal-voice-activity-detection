import os
from pathlib import Path
from typing import Dict, Optional, Union

import lhotse
from lhotse import load_manifest_lazy, CutSet, S3PRLSSLConfig, S3PRLSSL
from lhotse.recipes import prepare_switchboard


Pathlike = Union[Path, str]


def get_switchboard_cut(cut_set_path: Optional[Pathlike]) -> lhotse.CutSet:
    
    assert cut_set_path, f'cut_set_path must have a valid value. Currently, it is None'

    cut_set_path = Path(cut_set_path)
    assert cut_set_path.is_file(), f"No such file - cut_set_path: {cut_set_path}"

    return load_switchboard_cut(cut_set_path)


def load_switchboard_cut(cut_set_path: Pathlike) -> lhotse.CutSet:
    return load_manifest_lazy(cut_set_path)


def create_switchboard_dataset(
    audio_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    transcripts_dir: Optional[Pathlike] = None,
    **kwargs
) -> None:

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/switchboard/manifests')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    prepare_switchboard(
        audio_dir=audio_dir,
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        omit_silence=True,
        absolute_paths=True,
    )


# 1555 batches, 120 samples per batch, 5 seconds per sample
# 186505 samples in total
# 259.1666666666667 hours in total

def create_switchboard_cut(
    output_dir: Optional[Pathlike] = None,
    feats_dir: Optional[Pathlike] = None,
    batch_duration: int = 600,
    **kwargs
) -> lhotse.CutSet:

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/switchboard/manifests')
    feats_dir = Path(feats_dir) if feats_dir else Path('/export/c01/ashah108/vad/data/switchboard/feats')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    base_filename = 'swbd'

    rec = load_manifest_lazy(f'{output_dir}/swbd_recordings_all.jsonl.gz')
    sup = load_manifest_lazy(f'{output_dir}/swbd_supervisions_all.jsonl.gz')

    cuts = CutSet.from_manifests(
        recordings=rec,
        supervisions=sup
    )
    
    multi_cuts = cuts.multi_cuts
    mono_cuts = []
    for id, multi_cut in multi_cuts.items():
        mono_cuts.append(multi_cut.to_mono(mono_downmix=False)[0])
        
    cuts = CutSet.from_cuts(mono_cuts).resample(16000).cut_into_windows(duration=5).filter(lambda cut: cut.duration > 3)

    cuts.to_file(f'{output_dir}/{base_filename}_cuts_ssl.jsonl.gz')

    # extractor = Fbank(FbankConfig(
    #     sampling_rate=8000,
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

    cuts.to_file(f'{output_dir}/{base_filename}_all_cuts_feats_ssl.jsonl.gz')

    # cuts = CutSet.from_file(f'{output_dir}/{base_filename}_cuts_feats.jsonl.gz')
    ids = []
    for cut in cuts:
        ids.append(cut.id)

    num_samples = len(ids)

    train_cuts = cuts.subset(cut_ids=ids[:int(0.7 * num_samples)])
    dev_cuts = cuts.subset(cut_ids=ids[int(0.7 * num_samples):int(0.85 * num_samples)])
    test_cuts = cuts.subset(cut_ids=ids[int(0.85 * num_samples):])
    
    train_cuts.to_file(f'{output_dir}/{base_filename}_train_cuts_feats_ssl.jsonl.gz')
    dev_cuts.to_file(f'{output_dir}/{base_filename}_dev_cuts_feats_ssl.jsonl.gz')
    test_cuts.to_file(f'{output_dir}/{base_filename}_test_cuts_feats_ssl.jsonl.gz')

    return train_cuts, dev_cuts, test_cuts

