"""
Testing for custom collate fn

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
import json
import os
from pathlib import Path
import shutil

##third-party
import pandas as pd
import pytest

##local
from summer25.dataset import collate_features, collate_waveform
from summer25.models import HFExtractor

def load_audio():
    path = Path('./tests/audio_examples/')
    audio_files = path.rglob('*.flac')
    audio_files = [a for a in audio_files]

    wav1, sr = torchaudio.load(audio_files[0])
    sample1 = {'uid':'test1', 'waveform': wav1, 'sample_rate':sr, 'targets':[1.0,1.0,1.0,0.0]}
    
    wav2, sr = torchaudio.load(audio_files[1])
    sample2 = {'uid':'test2','waveform': wav2, 'sample_rate':sr, 'targets':[0.0,0.0,0.0,1.0]}

    return sample1, sample2 

def test_collate_features():
    sample1, sample2 = load_audio

    #whisper
    extract = HFExtractor(model_type='whisper-tiny')
    sample1 = extract(sample1)
    sample2 = extract(sample2)
    batch = [sample1, sample2]

    sample = collate_features(batch)
    assert 'uid' in sample
    assert sample['uid'] == ['test1','test2'], 'Batched incorrectly'
    assert 'sample_rate' in sample 
    assert sample['sample_rate'] == 16000
    assert 'targets' in sample
    assert sample['targets'] = [[1.0,1.0,1.0,0.0],[0.0,0.0,0.0,1.0]]
    assert 'waveform' in sample
    assert sample['waveform'].shape[0] == 2 and sample['waveform'].shape[1] == 80 and sample['waveform'].shape[2] == 300

    #try w batch size 1
    sample = collate_features([sample1])

    ##NON-WHISPER MODEL