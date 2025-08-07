"""
Test hugging face feature extractor

Author(s): Daniela Wiepert
Last modified: 08/2025
"""
#IMPORTS
import json
from pathlib import Path
import shutil

##third-party
import torchaudio
import pytest
from google.cloud import storage

##local
from summer25.models import HFExtractor

##### HELPER FUNCTIONS #####
def load_json():
    with open('./private_loading/gcs.json', 'r') as file:
        data = json.load(file)

    gcs_prefix = data['checkpoint_prefix']
    bucket_name = data['bucket_name']
    project_name = data['project_name']
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.get_bucket(bucket_name)
    return gcs_prefix, bucket

def load_audio():
    path = Path('./tests/audio_examples/')
    audio_files = path.rglob('*.flac')
    audio_files = [a for a in audio_files]

    wav1, sr = torchaudio.load(audio_files[0])
    sample1 = {'waveform': wav1, 'sample_rate':sr, 'targets':[1.0,1.0,1.0,0.0]}
    
    wav2, sr = torchaudio.load(audio_files[1])
    sample2 = {'waveform': wav2, 'sample_rate':sr, 'targets':[0.0,0.0,0.0,1.0]}

    return sample1, sample2 

##### TESTS #####
@pytest.mark.hf
def test_extractor_pretrained():
    #base test - load from hub
    m = HFExtractor(model_type='wavlm-base')
    assert m is not None, 'Extractor not running properly.'
    sample1, sample2 = load_audio()
    waveforms = [sample1['waveform'], sample2['waveform']]
    out, attn = m(waveforms)
    assert attn is not None, 'Extractor not running properly.'

    #invalid model type (not hugging face)
    with pytest.raises(AssertionError):
        m = HFExtractor(model_type='test_model')

    #test loading from hub but failure and tries for checkpoint
    with pytest.raises(AssertionError):
        m = HFExtractor(model_type='wavlm-base', test_local_fail=True)

    #not loading from hub but not given pt checkpoint
    # from hub false
    with pytest.raises(AssertionError):
        m = HFExtractor(model_type='wavlm-base', from_hub=False)

    #checkpoint named but doesn't exist
    #ckpt named but doesn't exist
    pt_ckpt = Path('./ckpt')
    if pt_ckpt.exists():
        shutil.rmtree(pt_ckpt)
    with pytest.raises(AssertionError):
        m = HFExtractor(model_type='wavlm-base', pt_ckpt=pt_ckpt, from_hub=False)

    #checkpoint created but has no models
    pt_ckpt.mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        m = HFExtractor(model_type='wavlm-base', pt_ckpt=pt_ckpt, from_hub=False)

    #defaults back to hub even if given checkpoint (no failure)
    m = HFExtractor(model_type='wavlm-base', pt_ckpt=pt_ckpt)
    assert m is not None, 'Extractor not running properly.'

    #loading local directory
    m = HFExtractor(model_type='wavlm-base', test_hub_fail=True)
    assert m is not None, 'Extractor not running properly.'
    assert m.local_path.exists(), 'Local path with copy of checkpoint does not exist.'

    #test exception if local failure and incompatible checkpoint
    with pytest.raises(ValueError):
        m = HFExtractor(model_type='wavlm-base', test_local_fail=True, pt_ckpt=pt_ckpt)

    #check no failure if given a true ckpt and fails
    pt_ckpt = m.local_path
    m = HFExtractor(model_type='wavlm-base', test_local_fail=True, pt_ckpt=pt_ckpt)
    assert m is not None, 'Extractor not running properly.'
    
    #check no failure if loading directly from checkpoint
    m = HFExtractor(model_type='wavlm-base', from_hub=False, pt_ckpt=pt_ckpt)
    assert m is not None, 'Extractor not running properly.'
    
    #check delete a checkpoint download
    m = HFExtractor(model_type='wavlm-base', test_hub_fail=True, delete_download=True)
    assert not m.local_path.exists(), 'Local path to checkpoint not deleted.'
    
    if pt_ckpt.exists():
        shutil.rmtree(pt_ckpt)

@pytest.mark.gcs
def test_extractor_pretrained_gcs():
    #base test - don't load from hub
    ckpt, bucket = load_json()
    params = {'model_type': 'wavlm-base', 'from_hub':False, 'pt_ckpt': ckpt, 'bucket':bucket, 'delete_download':True}
    m = HFExtractor(**params)
    assert m is not None, 'Extractor not running properly.'
    assert not m.local_path.exists(), 'Local path to checkpoint not deleted.'

    sample1, sample2 = load_audio()
    waveforms = [sample1['waveform'], sample2['waveform']]
    out, attn = m(waveforms)
    assert attn is not None, 'Extractor not running properly.'
    
    #CHECKPOINT THAT DOESN'T EXIST
    #invalid ckpt
    params['pt_ckpt'] = f'{ckpt}other'
    with pytest.raises(AssertionError):
        m = HFExtractor(**params)
