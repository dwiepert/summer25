"""
Test transforms

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
#built-in
from pathlib import Path
import shutil
#third-party
import numpy as np
import pytest
import torch
import torchaudio

#local
from summer25.transforms import *
from summer25.constants import _FEATURES

def load_audio():
    path = Path('/tests/audio_examples/')
    audio_files = path.rglob('*.flac')
    audio_files = [a for a in audio_files]

    wav1, sr = torchaudio.load(audio_files[0])
    sample1 = {'waveform': wav1, 'sample_rate':sr, 'targets':[1.0,1.0,1.0,0.0]}
    
    wav2, sr = torchaudio.load(audio_files[1])
    sample2 = {'waveform': wav2, 'sample_rate':sr, 'targets':[0.0,0.0,0.0,1.0]}

    return sample1, sample2
    

def test_mixup():
    sample1, sample2 = load_audio()
    m = Mixup()

    #try with only one input
    newsample = m(sample1=sample1)
    w1 = sample1['waveform']
    n1 = newsample['waveform']
    assert all([w1[i] == n1[i] for i in range(len(w1))]), 'Mixup did not return single sample.'

    #try with two
    newsample2 = m(sample1=sample1, sample2=sample2)
    w2 = sample2['waveform']
    n2 = newsample2['waveform']

    assert any([w1[i] != n2[i] for i in range(len(w1))]), 'Not mixed'
    assert any([w2[i] == n2[i] for i in range(len(w2))]), 'Not mixed'

def test_resample():
    sample1, _ = load_audio()
    # resample
    r = ResampleAudio(resample_rate=14000, librosa=False)
    resample1 = r(sample=sample1)
    assert resample1['sample_rate'] == 14000, 'Not new sample rate'
    assert sample1['sample_rate'] != resample1['sample_rate'], 'Not new sample rate'
    assert len(sample1['waveform']) > len(resample1['waveform']), 'Not new sample rate'

    #with librosa
    r2 = ResampleAudio(resample_rate=14000, librosa=True)
    resample2 = r2(sample=sample1)
    assert resample2['sample_rate'] == 14000, 'Not new sample rate'
    assert sample1['sample_rate'] != resample2['sample_rate'], 'Not new sample rate'
    assert len(sample1['waveform']) > len(resample2['waveform']), 'Not new sample rate'

    #sample sample rate
    r3 = ResampleAudio(resample_rate=sample1['sample_rate'], librosa=False)
    resample3 = r3(sample=sample1)
    assert resample3['sample_rate'] == sample1['sample_rate'], 'Not same sample rate.'
    assert sample1['sample_rate'] == resample2['sample_rate'], 'Not same sample rate.'
    assert len(sample1['waveform']) == len(resample2['waveform']), 'Not same waveform.'
    assert all([sample1['waveform'][i] == resample3['waveform'][i] for i in range(len(sample1['waveform']))]), 'Not same waveform.'

def test_monochannel():
    sample1, _ = load_audio()

    #test without actual multichannel
    m = ToMonophonic()
    msample1 = m(sample=sample1)
    assert msample1['waveform'].shape[0] == 1, 'Not monochannel'
    assert all([sample1['waveform'][i] == msample1['waveform'][i] for i in range(len(msample1['waveform']))]), 'Not equivalent to starting monochannel waveform'

    #test w multichannel
    temp =  sample1['waveform']
    temp = temp.unsqueeze(0)
    temp = temp.repeat(2,1)
    sample1['waveform'] = temp
    assert sample1['waveform'].shape[0] == 2, 'Not made stereo'
    msample2 = m(sample=sample1)
    assert msample2['waveform'].shape[0] == 1, 'Not monochannel'

    #summed sample1
    summed_wav = torch.sum(sample1['waveform'], axis=0).unsqueeze(0)
    assert all([summed_wav[i] == msample2['waveform'][i] for i in range(len(msample2['waveform']))])


def test_trimsilence():
    sample1, _ = load_audio()
    wav = sample1['waveform']
    sr = sample1['sample_rate']
    silence = torch.zeros((1,int(sr)))
    new_wav = torch.concat([silence,wav, silence], dim=1)
    sample1['waveform'] = new_wav

    t = TrimBeginningEndSilence()
    outsample = t(sample1)
    outwav = outsample['waveform']

    assert torch.equal(new_wav[0,:sr],outwav[0,:sr]) is False, 'No values removed from beginning.'
    assert torch.equal(new_wav[0,-sr:], outwav[0,-sr:]) is False, 'No values removed from end.'
    assert new_wav.shape[1] != outwav.shape[1], 'No values removed.'

def test_truncate():
    sample1, _ = load_audio()
    wav = sample1['waveform']
    sr = sample1['sample_rate']
    t = Truncate(length=10)
    outsample1 = t(sample1)
    outwav1 = outsample1['waveform']

    assert outwav1.shape[1] == int(sr*10), 'Not truncated to proper length.'
    assert torch.equal(outwav1, wav[0,:int(sr*10)]), 'Not truncated properly.'

    t2 = Truncate(length=10, offset=5)
    outsample2 = t2(sample1)
    outwav2 = outsample2['waveform']
    assert outwav2.shape[1] == int(sr*10), 'Not truncated to proper length.'
    frames = int(sr*10)
    offset = int(sr*5)
    assert torch.equal(wav[0,offset:offset+frames], outwav2), 'Not truncated properly with offset.'


def test_wavmean():
    sample1, _ = load_audio()
    wav = sample1['waveform']

    w = WaveMean()
    outsample = w(sample1)
    outwav = outsample['waveform']
    assert torch.equal(wav, outwav) is False, 'Did not change waveform.'

def test_tonumpy():
    sample1, _ = load_audio()
    t = ToNumpy()

    outsample = t(sample1)
    outwav = outsample['waveform']
    assert isinstance(outwav, np.ndarray), 'Not converted to numpy'

def test_totensor():
    sample1, _ = load_audio()
    sample1['targets'] = np.array([0.0,1.0,0.0,0.0])
    t = ToTensor()
    outsample = t(sample1)
    assert isinstance(outsample['waveform'],torch.Tensor), 'Did not convert targets to a torch tensor.'

def test_uid_to_path():
    sample1 = {'uid':'1919-142785-0008'}
    u = UidToPath(prefix=Path('/test/audio_examples/'), savedir=None, bucket=None, ext='flac', strcutured=False)

    outsample = u(sample1)
    assert outsample['waveform'] == '/test/audio_examples/1919-142785-0008.flac', 'Did not correctly convert to path.'

    #try prefix doesn't exist
    with pytest.raises(AssertionError):
         u = UidToPath(prefix=Path('/test/audio/'), savedir=None, bucket=None, ext='flac', structured=False)
    
    #try bucket stuff
    #TODO

    #try uid doesn't exist
    sample2 = {'uid':'1919-142785-00009'}
    u = UidToPath(prefix=Path('/test/audio_examples/'), savedir=None, bucket=None, ext='flac', structured=False)

    with pytest.raises(AssertionError):
        out = u(sample2)

    #try structured stuff
    uid = sample1['uid']
    temp_structured = Path(f'/tests/structured/{uid}')
    if not temp_structured.exists():
        temp_structured.mkdir(parents=True)
    shutil.copy(f'/tests/audio_examples/{uid}.flac', f'/tests/structured/{uid}/waveform.flac')
    u = UidToPath(prefix=Path('/test/audio_examples/'), savedir=None, bucket=None, ext='flac', structured=True)
    outsample = u(sample1)
    assert outsample['waveform'] == f'/test/structured/{uid}.flac', 'Did not correctly convert to path.'
    shutil.rmtree('/tests/structured')

    #bucket
    #TODO
 

def test_uid_to_waveform():
    #NO BUCKET
    #prefix doesn't exist
    sample1 = {'uid':'1919-142785-0008'}
    #unstructured
    with pytest.raises(AssertionError):
        u = UidToWaveform(prefix='/tests/audio/', bucket=None, extension = 'flac', lib=False, structured=False)

    u = UidToWaveform(prefix='/tests/audio_examples/', bucket=None, extension = 'flac', lib=False, structured=False)
    outsample = u(sample1)
    assert isinstance(outsample['waveform'], torch.Tensor), 'Waveform not loaded'
    assert outsample['sample_rate'] == 16000, 'Incorrectly loaded sample rate'

    #structured
    uid = sample1['uid']
    temp_structured = Path(f'/tests/structured/{uid}')
    if not temp_structured.exists():
        temp_structured.mkdir(parents=True)
    shutil.copy(f'/tests/audio_examples/{uid}.flac', f'/tests/structured/{uid}/waveform.flac')
    u = UidToWaveform(prefix='/tests/structured/', bucket=None, extension = 'flac', lib=False, structured=True)
    outsample = u(sample1)
    assert isinstance(outsample['waveform'], torch.Tensor), 'Waveform not loaded'
    assert outsample['sample_rate'] == 16000, 'Incorrectly loaded sample rate'
    shutil.rmtree('/tests/structured')

    #librosa
    u = UidToWaveform(prefix='/tests/audio_examples/', bucket=None, extension = 'flac', lib=True, structured=False)
    outsample = u(sample1)
    assert isinstance(outsample['waveform'],  torch.Tensor), 'Waveform not loaded'
    assert outsample['sample_rate'] == 16000, 'Incorrectly loaded sample rate'
   
    #BUCKET - unstructured and structured, librosa
    #TODO
