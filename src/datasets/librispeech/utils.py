import os
from pathlib import Path
from typing import Dict, Optional, Union

import lhotse
from lhotse import load_manifest_lazy, FbankConfig, Fbank, CutSet
from lhotse.recipes import prepare_librispeech


Pathlike = Union[Path, str]


def get_librispeech_cut(
    cut_set_path: Optional[Pathlike] = None,
    audio_dir: Optional[Pathlike] = None,
    output_dir: Optional[Pathlike] = None,
    feats_dir: Optional[Pathlike] = None,
    batch_duration: int = 600,
    phase: Optional[str] = 'train',
    prepare_dataset: Optional[bool] = False,
    get_cuts: Optional[bool] = True
) -> lhotse.CutSet:
    
    assert cut_set_path or audio_dir, f'One of "cut_set_path" or "audio_dir" must have a valid value. \
                                        Currently, both are None'

    if cut_set_path:
        cut_set_path = Path(cut_set_path)
        assert cut_set_path.is_file(), f"No such file - cut_set_path: {cut_set_path}"

        return load_librispeech_cut(cut_set_path)

    else:
        audio_dir = Path(audio_dir)
        assert audio_dir.is_dir(), f"No such directory - audio_dir: {audio_dir}"

        return create_librispeech_cut(
            audio_dir,
            output_dir,
            feats_dir,
            batch_duration,
            phase,
            prepare_dataset,
            get_cuts
        )

def load_librispeech_cut(cut_set_path: Pathlike) -> lhotse.CutSet:
    return load_manifest_lazy(cut_set_path)


def create_librispeech_cut(
    audio_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    feats_dir: Optional[Pathlike] = None,
    batch_duration: int = 600,
    phase: Optional[str] = 'train',
    prepare_dataset: Optional[bool] = False,
    get_cuts: Optional[bool] = True
) -> lhotse.CutSet:

    hrs = 360

    output_dir = Path(output_dir) if output_dir else Path('/export/c01/ashah108/vad/data/librispeech/manifests')
    feats_dir = Path(feats_dir) if feats_dir else Path('/export/c01/ashah108/vad/data/librispeech/feats')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(feats_dir):
        feats_dir.mkdir(parents=True, exist_ok=True)

    if prepare_dataset:
        prepare_librispeech(corpus_dir=audio_dir, output_dir=output_dir)
    
    if not get_cuts:
        return None

    if phase == 'train':
        rec = load_manifest_lazy(f'{output_dir}/librispeech_recordings_train-clean-{hrs}.jsonl.gz')
        sup = load_manifest_lazy(f'{output_dir}/librispeech_supervisions_train-clean-{hrs}.jsonl.gz')
        base_filename = f'{phase}_{hrs}'

    else:  # dev or test
        rec = load_manifest_lazy(f'{output_dir}/librispeech_recordings_{phase}-clean.jsonl.gz')
        sup = load_manifest_lazy(f'{output_dir}/librispeech_supervisions_{phase}-clean.jsonl.gz')
        base_filename = f'{phase}' 

    cuts = CutSet.from_manifests(
        recordings=rec,
        supervisions=sup
    ).cut_into_windows(duration=5).filter(lambda cut: cut.duration > 3)

    cuts.to_file(f'{output_dir}/{base_filename}_cuts.jsonl.gz')

    extractor = Fbank(FbankConfig(device='cuda'))

    cuts = cuts.compute_and_store_features_batch(
        extractor=extractor,
        storage_path=f'{feats_dir}/{base_filename}_feats',
        batch_duration=batch_duration,
        num_workers=4,
        overwrite=True
    ).pad(duration=5.0)


    cuts.to_file(f'{output_dir}/{base_filename}_cuts_feats.jsonl.gz')
    # cuts.describe()

    return cuts

