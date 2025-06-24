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
import torchaudio
import torch

##local
from summer25.dataset import collate_features, collate_waveform
from summer25.models import HFExtractor

def load_audio():
    path = Path('./tests/audio_examples/')
    audio_files = path.rglob('*.flac')
    audio_files = [a for a in audio_files]

    wav1, sr = torchaudio.load(audio_files[0])
    sample1 = {'uid':'test1', 'waveform': wav1, 'sample_rate':sr, 'targets':torch.tensor([1.0,1.0,1.0,0.0])}
    
    wav2, sr = torchaudio.load(audio_files[1])
    sample2 = {'uid':'test2','waveform': wav2, 'sample_rate':sr, 'targets':torch.tensor([0.0,0.0,0.0,1.0])}

    return sample1, sample2 

def test_collate_features():
    sample1, sample2 = load_audio()

    #whisper
    extract = HFExtractor(model_type='whisper-tiny')
    sample1 = extract(sample1)
    sample2 = extract(sample2)
    batch = [sample1, sample2]

    sample = collate_features(batch)
    assert 'uid' in sample
    assert sample['uid'] == ['test1','test2'], 'Batched incorrectly'
    assert 'sample_rate' in sample 
    assert sample['sample_rate'] == [16000,16000]
    assert 'targets' in sample
    assert sample['targets'].shape[0] == 2 and sample['targets'].shape[1] == 4 
    assert 'waveform' in sample
    assert sample['waveform'].shape[0] == 2 and sample['waveform'].shape[1] == 80 and sample['waveform'].shape[2] == 3000

    #try w batch size 1
    sample = collate_features([sample1])
    assert sample['waveform'].shape[0] == 1 and sample['waveform'].shape[1] == 80 and sample['waveform'].shape[2] == 3000

def test_collate_waveform():
    sample1, sample2 = load_audio()

    #wavlm-base
    extract = HFExtractor(model_type='wavlm-base')
    sample1 = extract(sample1)
    sample2 = extract(sample2)
    batch = [sample1, sample2]

    sample = collate_waveform(batch, pad=False)
    assert 'uid' in sample
    assert sample['uid'] == ['test1','test2'], 'Batched incorrectly'
    assert 'sample_rate' in sample 
    assert sample['sample_rate'] == [16000,16000]
    assert 'targets' in sample
    assert sample['targets'].shape[0] == 2 and sample['targets'].shape[1] == 4 
    assert 'waveform' in sample
    assert isinstance(sample['waveform'],list)


    sample = collate_waveform(batch, pad=True, pad_method='mean')
    assert 'uid' in sample
    assert sample['uid'] == ['test1','test2'], 'Batched incorrectly'
    assert 'sample_rate' in sample 
    assert sample['sample_rate'] == [16000,16000]
    assert 'targets' in sample
    assert sample['targets'].shape[0] == 2 and sample['targets'].shape[1] == 4 
    assert 'waveform' in sample
    outwaveform = sample['waveform']
    max_len = max([b['waveform'].shape[1] for b in batch])
    assert outwaveform.shape[0] == 2 and outwaveform.shape[1] == max_len and outwaveform[1,-1] != 0
    
    sample = collate_waveform(batch, pad=True, pad_method='zero')
    assert 'uid' in sample
    assert sample['uid'] == ['test1','test2'], 'Batched incorrectly'
    assert 'sample_rate' in sample 
    assert sample['sample_rate'] == [16000,16000]
    assert 'targets' in sample
    assert sample['targets'].shape[0] == 2 and sample['targets'].shape[1] == 4 
    assert 'waveform' in sample
    outwaveform = sample['waveform']
    max_len = max([b['waveform'].shape[1] for b in batch])
    assert outwaveform.shape[0] == 2 and outwaveform.shape[1] == max_len and outwaveform[1,-1] == 0
    
    #TODO: try with batch size 1
    sample = collate_waveform([sample1], pad=True, pad_method='zero')
    outwaveform = sample['waveform']
    assert outwaveform.shape[0] == 1 and outwaveform.shape[1] == sample1['waveform'].shape[1]

    sample = collate_waveform([sample1], pad=False)
    outwaveform = sample['waveform']
    assert outwaveform.shape[0] == 1 and outwaveform.shape[1] == sample1['waveform'].shape[1]