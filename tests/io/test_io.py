"""
Test I/O functions

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
import shutil
##third-party
import pytest
from pathlib import Path
import torch
##local
from summer25.io import load_waveform_from_local

@pytest.mark.gcs
def test_download_to_local():
    pass

def test_load_local():
    #not structured
    input_dir=Path('./tests/audio_examples/')
    uid = '1919-142785-0008'
    extension = 'flac'
    waveform, sr = load_waveform_from_local(input_dir=input_dir, uid=uid,extension=extension, lib=False, structured=False)
    assert isinstance(waveform, torch.Tensor), 'Waveform not loaded'
    assert sr == 16000, 'Incorrectly loaded sample rate'

    #librosa
    waveform, sr = load_waveform_from_local(input_dir=input_dir, uid=uid,extension=extension, lib=True, structured=False)
    assert isinstance(waveform, torch.Tensor), 'Waveform not loaded'
    assert sr == 16000, 'Incorrectly loaded sample rate'

    #not existing audio 
    with pytest.raises(FileNotFoundError):
        uid = '1919-142785-0009'
        waveform, sr = load_waveform_from_local(input_dir=input_dir, uid=uid,extension=extension, lib=False, structured=False)
    
    #structured
    uid = '1919-142785-0008'
    temp_structured = Path(f'./tests/structured/{uid}')
    if temp_structured.exists():
        shutil.rmtree(temp_structured)
    if not temp_structured.exists():
        temp_structured.mkdir(parents=True)
    shutil.copy(f'./tests/audio_examples/{uid}.flac', f'./tests/structured/{uid}/waveform.flac')

    waveform, sr = load_waveform_from_local(input_dir=Path('./tests/structured'), uid=uid,extension=extension, lib=False, structured=True)
    assert isinstance(waveform, torch.Tensor), 'Waveform not loaded'
    assert sr == 16000, 'Incorrectly loaded sample rate'
    shutil.rmtree('./tests/structured')

@pytest.mark.gcs
def test_load_gcs():
    pass