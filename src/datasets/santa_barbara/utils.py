import os
from pathlib import Path
from typing import Dict, Optional, Union

import lhotse
from lhotse import load_manifest_lazy, CutSet, Fbank, FbankConfig, S3PRLSSLConfig, S3PRLSSL
from lhotse.supervision import SupervisionSegment, SupervisionSet


Pathlike = Union[Path, str]


def get_santa_barbara_cut(cut_set_path: Optional[Pathlike]) -> lhotse.CutSet:
    
    assert cut_set_path, f'cut_set_path must have a valid value. Currently, it is None'

    cut_set_path = Path(cut_set_path)
    assert cut_set_path.is_file(), f"No such file - cut_set_path: {cut_set_path}"

    return load_santa_barbara_cut(cut_set_path)

def load_santa_barbara_cut(cut_set_path: Pathlike) -> lhotse.CutSet:
    return load_manifest_lazy(cut_set_path)


def create_santa_barbara_dataset(
    output_dir: Optional[Pathlike] = None,
    **kwargs
) -> None:

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/santa_barbara/manifests')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    rec = load_manifest_lazy(f'{output_dir}/sbcsae_rec.jsonl.gz')
    supervisions = []

    for r in rec:
        with open(f'/export/c01/ashah108/vad/src/datasets/santa_barbara/labels/{r.id}_autoaligned_nonfa_iol_v3.txt', 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.strip().split('\t')
                start = float(line[0])
                end = float(line[1])
                text = line[2]
                supervisions.append(
                    SupervisionSegment(
                        id=f"{r.id}-{idx}",
                        recording_id=r.id,
                        start=round(start, 2),
                        duration=round(end-start, 2),
                        channel=0,
                        text=text,
                        language="English",
                    )
                )


    supervisions = SupervisionSet.from_segments(supervisions)
    supervisions.to_file(f'{output_dir}/sbcsae_supervisions.jsonl.gz')


def create_santa_barbara_cut(
    output_dir: Optional[Pathlike] = None,
    feats_dir: Optional[Pathlike] = None,
    batch_duration: int = 600,
    feature_extractor: Optional[str] = 'wav2vec2',
    **kwargs
) -> lhotse.CutSet:

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/santa_barbara/manifests')
    feats_dir = Path(feats_dir) if feats_dir else Path('/export/c01/ashah108/vad/data/santa_barbara/feats')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    rec = load_manifest_lazy(f'{output_dir}/sbcsae_rec.jsonl.gz')
    sup = load_manifest_lazy(f'{output_dir}/sbcsae_supervisions.jsonl.gz')
    base_filename = 'santa_barbara'

    cuts = CutSet.from_manifests(
        recordings=rec,
        supervisions=sup
    )

    cuts = cuts.resample(16000)
    
    cuts.to_file(f'{output_dir}/{base_filename}_cuts_og.jsonl.gz')

    # cuts_trim = cuts.trim_to_supervisions(keep_overlapping=False, keep_all_channels=False)
    # cuts_trim.to_file(f'{output_dir}/{base_filename}_cuts_trim.jsonl.gz')
    
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


    cuts.to_file(f'{output_dir}/{base_filename}_cuts_{feature_extractor}.jsonl.gz')
    # cuts.describe()

    return cuts

