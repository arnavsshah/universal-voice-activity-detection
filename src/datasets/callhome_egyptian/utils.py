import os
from pathlib import Path
from typing import Dict, Optional, Union

import lhotse
from lhotse import load_manifest_lazy, CutSet, S3PRLSSLConfig, S3PRLSSL
from lhotse.recipes import prepare_callhome_egyptian


Pathlike = Union[Path, str]


def get_callhome_egyptian_cut(cut_set_path: Optional[Pathlike]) -> lhotse.CutSet:
    
    assert cut_set_path, f'cut_set_path must have a valid value. Currently, it is None'

    cut_set_path = Path(cut_set_path)
    assert cut_set_path.is_file(), f"No such file - cut_set_path: {cut_set_path}"

    return load_callhome_egyptian_cut(cut_set_path)

def load_callhome_egyptian_cut(cut_set_path: Pathlike) -> lhotse.CutSet:
    return load_manifest_lazy(cut_set_path)


def create_callhome_egyptian_cut(
    audio_dir: Pathlike,
    transcript_dir: Optional[Pathlike] = None,
    output_dir: Optional[Pathlike] = None,
    feats_dir: Optional[Pathlike] = None,
    batch_duration: int = 600,
    phase: Optional[str] = 'train',
    prepare_dataset: Optional[bool] = False,
    get_cuts: Optional[bool] = True
) -> lhotse.CutSet:

    phases = ["evaltest", "train", "devtest"]
    assert phase in phases, f'Invalid phase: {phase}. Phase should be in {phases}'

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/callhome_egyptian/manifests')
    feats_dir = Path(feats_dir) if feats_dir else Path('/export/c01/ashah108/vad/data/callhome_egyptian/feats')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    if prepare_dataset:
        prepare_callhome_egyptian(
            audio_dir=audio_dir,
            transcript_dir=transcript_dir,
            output_dir=output_dir,
            absolute_paths=True,
        )

    if not get_cuts:
        return None

    else:
        rec = load_manifest_lazy(f'{output_dir}/callhome-egyptian_recordings_{phase}.jsonl.gz')
        sup = load_manifest_lazy(f'{output_dir}/callhome-egyptian_supervisions_{phase}.jsonl.gz')
        base_filename = f'{phase}'

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


        cuts.to_file(f'{output_dir}/{base_filename}_cuts_feats_ssl.jsonl.gz')
        # cuts.describe()

        return cuts

