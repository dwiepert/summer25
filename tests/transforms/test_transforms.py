"""
Test transforms

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
#built-in
from pathlib import Path
import shutil
import os 

#third-party
import numpy as np
import pytest
import torch
import torchaudio

#local
from summer25.transforms import *
from summer25.constants import _FEATURES

def load_audio():
    path = Path('./tests/audio_examples/')
    audio_files = path.rglob('*.flac')
    audio_files = [a for a in audio_files]

    wav1, sr = torchaudio.load(audio_files[0])
    sample1 = {'waveform': wav1, 'sample_rate':sr, 'targets':[1.0,1.0,1.0,0.0]}
    
    wav2, sr = torchaudio.load(audio_files[1])
    sample2 = {'waveform': wav2, 'sample_rate':sr, 'targets':[0.0,0.0,0.0,1.0]}

    return sample1, sample2 

def test_resample():
    sample1, _ = load_audio()
    # resample
    r = ResampleAudio(resample_rate=14000)
    resample1 = r(sample=sample1)
    assert resample1['sample_rate'] == 14000, 'Not new sample rate'
    assert sample1['sample_rate'] != resample1['sample_rate'], 'Not new sample rate'
    assert sample1['waveform'].shape[1] > resample1['waveform'].shape[1], 'Not new sample rate'

    #same sample rate
    r2 = ResampleAudio(resample_rate=sample1['sample_rate'])
    resample2 = r2(sample=sample1)
    assert resample2['sample_rate'] == sample1['sample_rate'], 'Not same sample rate.'
    assert sample1['waveform'].shape[1] == resample2['waveform'].shape[1], 'Not same waveform.'
    assert torch.equal(sample1['waveform'],resample2['waveform']), 'Not same waveform.'

def test_monochannel():
    sample1, _ = load_audio()

    #test without actual multichannel
    m = ToMonophonic()
    msample1 = m(sample=sample1)
    assert msample1['waveform'].shape[0] == 1, 'Not monochannel'
    assert torch.equal(sample1['waveform'], msample1['waveform']), 'Not equivalent to starting monochannel waveform'

    #test w multichannel
    temp =  sample1['waveform']
    temp = temp.repeat(2,1)
    sample1['waveform'] = temp
    assert sample1['waveform'].shape[0] == 2, 'Not made stereo'
    msample2 = m(sample=sample1)
    assert msample2['waveform'].shape[0] == 1, 'Not monochannel'

    #summed sample1
    summed_wav = torch.sum(sample1['waveform'], axis=0).unsqueeze(0)
    assert torch.equal(summed_wav, msample2['waveform']), 'Monochannel function not operating correctly'

    #test with reduce fn that doesn't work properly
    fn = lambda w: w
    m = ToMonophonic(reduce_fn=fn)
    with pytest.raises(ValueError):
        _ = m(sample=sample1)

def test_trimsilence():
    sample1, _ = load_audio()
    wav = sample1['waveform']
    sr = sample1['sample_rate']
    silence = torch.zeros((1,int(sr)))
    new_wav = torch.concat([silence,wav, silence], dim=1)
    sample1['waveform'] = new_wav

    t = TrimBeginningEndSilence(use_librosa=True)
    outsample = t(sample1)
    outwav = outsample['waveform']

    assert torch.equal(new_wav[0,:sr],outwav[0,:sr]) is False, 'No values removed from beginning.'
    assert torch.equal(new_wav[0,-sr:], outwav[0,-sr:]) is False, 'No values removed from end.'
    assert new_wav.shape[1] != outwav.shape[1], 'No values removed.'

    t = TrimBeginningEndSilence(use_librosa=False)
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
    assert torch.equal(outwav1, wav[:,:int(sr*10)]), 'Not truncated properly.'

    t2 = Truncate(length=10, offset=5)
    outsample2 = t2(sample1)
    outwav2 = outsample2['waveform']
    assert outwav2.shape[1] == int(sr*10), 'Not truncated to proper length.'
    frames = int(sr*10)
    offset = int(sr*5)
    assert torch.equal(wav[:,offset:offset+frames], outwav2), 'Not truncated properly with offset.'

    #TODO: PAD version - mean
    new_wav = wav[:,:frames-sr]
    sample1['waveform'] = new_wav
    t3 = Truncate(length=10, pad=True)
    out = t3(sample1)
    outwav3 = out['waveform']
    assert outwav3.shape[1] == frames, 'Not padded properly'
    assert float(outwav3[:,-1]) != 0, 'Not padded properly' 

    # PAD version zeros
    t4 = Truncate(length=10, pad=True, pad_method='zero')
    out = t4(sample1)
    outwav4 = out['waveform']
    assert outwav4.shape[1] == frames, 'Not padded properly'
    assert float(outwav4[:,-1]) == 0, 'Not padded properly' 

    #TODO: not pad short version
    t5 = Truncate(length=10, pad=False)
    out = t5(sample1)
    outwav5 = out['waveform']
    assert outwav5.shape[1] < frames, 'Not truncated properly'
    assert outwav5.shape[1] == new_wav.shape[1], 'Not truncated properly'

    with pytest.raises(AssertionError):
        t = Truncate(length=10, pad=True, pad_method='other')

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
    prefix = Path('./tests/audio_examples/')
    u = UidToPath(prefix=prefix, savedir=None, bucket=None, ext='flac', structured=False)

    outsample = u(sample1)
    assert '/tests/audio_examples/1919-142785-0008.flac' in outsample['waveform'], 'Did not correctly convert to path.'

    #try prefix doesn't exist
    with pytest.raises(AssertionError):
            u = UidToPath(prefix=Path('/tests/audio/'), savedir=None, bucket=None, ext='flac', structured=False)

    #try uid doesn't exist
    sample2 = {'uid':'1919-142785-00009'}
    u = UidToPath(prefix=prefix, savedir=None, bucket=None, ext='flac', structured=False)

    with pytest.raises(AssertionError):
        out = u(sample2)

    #try structured stuff
    uid = sample1['uid']
    temp_structured = Path(f'./tests/structured/{uid}')
    if temp_structured.exists():
        shutil.rmtree(temp_structured)
    temp_structured.mkdir(parents=True)
    shutil.copy(f'./tests/audio_examples/{uid}.flac', f'./tests/structured/{uid}/waveform.flac')
    u = UidToPath(prefix=Path('./tests/structured/'), savedir=None, bucket=None, ext='flac', structured=True)
    outsample = u(sample1)
    assert f'/tests/structured/{uid}/waveform.flac' in outsample['waveform'], 'Did not correctly convert to path.'
    shutil.rmtree('./tests/structured')

@pytest.mark.gcs
def test_uid_to_path_gcs():
    #bucket
    #TODO
    pass

def test_uid_to_waveform():
    sample1 = {'uid':'1919-142785-0008'}
    #unstructured
    with pytest.raises(AssertionError):
        u = UidToWaveform(prefix='/tests/audio/', bucket=None, extension = 'flac', lib=False, structured=False)

    u = UidToWaveform(prefix='./tests/audio_examples/', bucket=None, extension = 'flac', lib=False, structured=False)
    outsample = u(sample1)
    assert isinstance(outsample['waveform'], torch.Tensor), 'Waveform not loaded'
    assert outsample['sample_rate'] == 16000, 'Incorrectly loaded sample rate'

    #structured
    uid = sample1['uid']
    temp_structured = Path(f'./tests/structured/{uid}')
    if temp_structured.exists():
        shutil.rmtree(temp_structured)
    if not temp_structured.exists():
        temp_structured.mkdir(parents=True)
    shutil.copy(f'./tests/audio_examples/{uid}.flac', f'./tests/structured/{uid}/waveform.flac')
    u = UidToWaveform(prefix='./tests/structured/', bucket=None, extension = 'flac', lib=False, structured=True)
    outsample = u(sample1)
    assert isinstance(outsample['waveform'], torch.Tensor), 'Waveform not loaded'
    assert outsample['sample_rate'] == 16000, 'Incorrectly loaded sample rate'
    shutil.rmtree('./tests/structured')

    #librosa
    u = UidToWaveform(prefix='./tests/audio_examples/', bucket=None, extension = 'flac', lib=True, structured=False)
    outsample = u(sample1)
    assert isinstance(outsample['waveform'],  torch.Tensor), 'Waveform not loaded'
    assert outsample['sample_rate'] == 16000, 'Incorrectly loaded sample rate'

@pytest.mark.gcs
def test_uid_to_waveform_gcs():
    #BUCKET - unstructured and structured, librosa
    #TODO
    pass

def test_pad():
    sample1, _ = load_audio()
    with pytest.raises(AssertionError):
        p = Pad(pad_method='other')

    max_len = sample1['waveform'].shape[1] + 1000
    p = Pad(pad_method='mean')
    outsample = p(sample1, max_len=max_len)
    assert outsample['waveform'].shape[1] == max_len, 'Not padded to proper len'
    #assert outsample['waveform'][-1000:]

    p2 = Pad(pad_method='zero')
    outsample2 = p2(sample1, max_len=max_len)
    assert outsample2['waveform'].shape[1] == max_len, 'Not padded to proper len'
    assert torch.equal(outsample2['waveform'][:,-1000:], torch.zeros((1,1000))), 'Not padded with proper values.'

