"""
Test hugging face models

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
import os
from pathlib import Path
import shutil

##third-party
import pytest
import torchaudio

##local
from summer25.models import HFModel, Classifier, HFExtractor


@pytest.mark.hf
def test_hfmodel_pretrained_base():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)
    assert m is not None, 'Model not running properly.'

    #invalid model type (not hugging face)
    params['model_type']='test_model'
    with pytest.raises(AssertionError):
        m = HFModel(**params)
    
    #not loading from hub but not given pt ckpt
    params['model_type'] = 'wavlm-base'
    params['from_hub'] = False
    with pytest.raises(AssertionError):
        m = HFModel(**params)
    
    params['from_hub'] = True
    with pytest.raises(AssertionError):
        m = HFModel(test_local_fail=True, **params)

    params['from_hub'] = False
    #ckpt named but doesn't exist
    pt_ckpt = Path('./ckpt')
    if pt_ckpt.exists():
        shutil.rmtree(pt_ckpt)
    params['pt_ckpt'] = pt_ckpt
    with pytest.raises(AssertionError):
        m = HFModel(**params)

    #checkpoint created but has no models
    pt_ckpt.mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        m = HFModel(**params)
    
    #confirm it defaults back to the hub even if given checkpoint (no failure)
    params['from_hub'] = True
    m = HFModel(**params)
    assert m is not None, 'Model not running properly.'

    #test loading local directory
    m = HFModel(test_hub_fail=True, **params) #should work no issue - DON'T DELETE
    assert m is not None, 'Model not running properly.'
    assert m.local_path.exists(), 'Local path with copy of checkpoint does not exist.'
    
    #test exception raised if local failure and incompatible checkpoint
    pt_ckpt = m.local_path
    with pytest.raises(ValueError):
        m = HFModel(test_local_fail=True, **params)

    #check no failure if given a true ckpt and fails
    params['pt_ckpt'] = m.local_path
    m = HFModel(test_local_fail=True, **params)
    assert m is not None, 'Model not running properly.'

    #check no failure if loading directly from checkpoint
    params['from_hub'] = False
    m = HFModel(**params)
    assert m is not None, 'Model not running properly.'

    #check that you can delete a checkpoint download
    params['from_hub'] = True
    params['delete_download'] = True
    del params['pt_ckpt'] 

    m = HFModel(test_hub_fail=True, **params)
    assert not m.local_path.exists(), 'Local path to checkpoint not deleted.'
    shutil.rmtree(params['out_dir'])
    if pt_ckpt.exists():
        shutil.rmtree(pt_ckpt)
    
def test_hfmodel_finetuned_base():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'

    ### FINETUNED CHECKPOINT
    ft_ckpt = params['out_dir'] / 'weights'
    if ft_ckpt.exists():
        shutil.rmtree(ft_ckpt)
    params['ft_ckpt'] = ft_ckpt

    # doesn't exist
    with pytest.raises(AssertionError):
        m = HFModel(**params)

    #exists but no files
    ft_ckpt.mkdir(parents=True, exist_ok=True)
    with pytest.raises(AssertionError):
        m = HFModel(**params)
    
    del params['ft_ckpt']

    # check SAVING - NOT LORA
    m = HFModel(**params)
    m.save_model_components()
    assert (ft_ckpt/(m.model_name)).exists() and os.listdir(ft_ckpt/(m.model_name)) != [], 'Base model properly saved.'
    assert (ft_ckpt/(m.clf.config['clf_name'] + '.pt')).exists(), 'Classifier properly saved.'

    #FT POSSIBILITIES:
    # give path to only base_model directory (no classifier ckpt in dir, no subdirs)
    model_ft_ckpt = ft_ckpt/(m.model_name)
    params['ft_ckpt'] = model_ft_ckpt
    m = HFModel(**params)

    # give path to only clf ckpt (no ft_ckpt, just uploads from hub)
    del params['ft_ckpt']
    params['clf_ckpt'] = ft_ckpt/(m.config['clf_name'] + '.pt')
    m = HFModel(**params)

    # give path to both ft_ckpt (parent_dir) and clf_ckpt (use given clf_ckpt)
    params['ft_ckpt'] = ft_ckpt
    m = HFModel(**params)

    # give path to both ft_ckpt (actual dir) and clf_ckpt
    params['ft_ckpt'] = model_ft_ckpt
    m = HFModel(**params)

    # give path to parent dir for ft_ckpt and no clf_ckpt
    del params['clf_ckpt']
    params['ft_ckpt'] = ft_ckpt
    m = HFModel(**params)

    # Incompatible ft_ckpt
    params['model_type'] = 'whisper-base'
    with pytest.raises(ValueError):
        m = HFModel(**params)

    # Incompatible ft_ckpt
    params['model_type'] = 'hubert-base'
    with pytest.raises(ValueError):
        m = HFModel(**params)
    
    shutil.rmtree(params['out_dir'])
    if ft_ckpt.exists():
        shutil.rmtree(ft_ckpt)

def test_lora():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    params['finetune_method'] = 'lora'
    ft_ckpt = params['out_dir'] / 'weights'

    m = HFModel(**params)
    m.save_model_components()
    assert (ft_ckpt/(m.model_name)).exists() and os.listdir(ft_ckpt/(m.model_name)) != [], 'Base model properly saved.'
    assert (ft_ckpt/(m.clf.config['clf_name'] + '.pt')).exists(), 'Classifier properly saved.'

    #from hub = True 
    params['ft_ckpt'] = ft_ckpt/(m.model_name)
    m = HFModel(**params)

    params['from_hub'] = False 
    with pytest.raises(AssertionError):
        m = HFModel(**params)

    params['from_hub'] = True
    m = HFModel(test_hub_fail=True, **params) #should work no issue - DON'T DELETE
    assert m is not None, 'Model not running properly.'
    assert m.local_path.exists(), 'Local path with copy of checkpoint does not exist.'
    params['pt_ckpt'] = m.local_path 

    params['from_hub'] = False
    m = HFModel(**params)

    params['delete_download'] = True
    m = HFModel(test_hub_fail=True, **params)
    
    shutil.rmtree(params['out_dir'])
    if ft_ckpt.exists():
        shutil.rmtree(ft_ckpt)

def test_softprompt():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    params['finetune_method'] = 'soft-prompt'
    ft_ckpt = params['out_dir'] / 'weights'

    #TEST IT WORKS LOADING FROM HUB
    m = HFModel(**params)
    m.save_model_components()
    assert (ft_ckpt/(m.model_name)).exists() and os.listdir(ft_ckpt/(m.model_name)) != [], 'Base model properly saved.'
    assert (ft_ckpt/(m.clf.config['clf_name'] + '.pt')).exists(), 'Classifier properly saved.'

    #TEST IT ALSO WORKS LOADING FROM LOCAL PT CKPT
    params['from_hub'] = True
    m = HFModel(test_hub_fail=True, **params) #should work no issue - DON'T DELETE
    assert m is not None, 'Model not running properly.'
    assert m.local_path.exists(), 'Local path with copy of checkpoint does not exist.'
    params['pt_ckpt'] = m.local_path 

    params['from_hub'] = False
    m = HFModel(**params)   

    #from hub = True AND FT checkpoint already exists
    params['ft_ckpt'] = ft_ckpt/(m.model_name)
    m = HFModel(**params)

    params['from_hub'] = False 
    del params['pt_ckpt']
    with pytest.raises(AssertionError):
        m = HFModel(**params)

    params['pt_ckpt'] = m.local_path 
    m = HFModel(**params)

    params['delete_download'] = True
    m = HFModel(test_hub_fail=True, **params)
    
    shutil.rmtree(params['out_dir'])
    if ft_ckpt.exists():
        shutil.rmtree(ft_ckpt)

def check_requires_grad(model):
    unfrozen = model.unfreeze
    check = []
    for name, param in model.base_model.named_parameters():
        if any([f in name for f in unfrozen]):
            check.append(param.requires_grad)
        else:
            check.append(param.requires_grad is False)
    return all(check)

@pytest.mark.hf
def test_freeze():
    params = {'out_dir':Path('./out_dir'), 'model_type':'wavlm-base'}

    #freeze all
    #no freezing (required only)
    params['freeze_method'] = 'all'
    m = HFModel(**params)
    assert m is not None, 'Model not running properly.'
    assert check_requires_grad(m)

    #no freezing (required only)
    params['freeze_method'] = 'required-only'
    m = HFModel(**params)
    assert m is not None, 'Model not running properly.'
    assert check_requires_grad(m)

    #freeze optional
    params['freeze_method'] = 'optional'
    m1 = HFModel(**params)
    assert m1 is not None, 'Model not running properly.'
    assert check_requires_grad(m1)

    #freeze half
    params['freeze_method'] = 'half'
    m2 = HFModel(**params)
    assert m2 is not None, 'Model not running properly.'
    assert check_requires_grad(m2)
    assert len(m2.unfreeze) == 6

    #freeze all but last
    params['freeze_method'] = 'exclude-last'
    m3 = HFModel(**params)
    assert m3 is not None, 'Model not running properly.'
    assert check_requires_grad(m3)
    assert len(m3.unfreeze) == 1

    #freeze specific layers
    params['freeze_method'] = 'layer'
    params['unfreeze_layers'] = ['encoder.layers.5']
    m4 = HFModel(**params)
    assert m4 is not None, 'Model not running properly.'
    assert check_requires_grad(m4)
    assert len(m4.unfreeze) == 1 and m4.unfreeze == params['unfreeze_layers']

    #TRY WITH HUBERT
    params['model_type'] = 'hubert-base'
    params['freeze_method'] = 'optional'
    del params['unfreeze_layers']
    m5 = HFModel(**params)

    #TRY WITH WHISPER
    params['model_type'] = 'whisper-tiny'
    params['freeze_method'] = 'optional'
    m6 = HFModel(**params)
    assert m6 is not None, 'Model not running properly.'
    assert check_requires_grad(m6)

    #Check whisper with half and exclude last 
    params['freeze_method'] = 'half'
    m7 = HFModel(**params)
    assert m7 is not None, 'Model not running properly.'
    assert check_requires_grad(m7)
    assert len(m7.unfreeze) == 3 

    params['freeze_method'] = 'exclude-last'
    m8 = HFModel(**params)
    assert m8 is not None, 'Model not running properly.'
    assert check_requires_grad(m8)
    assert len(m8.unfreeze) == 2

    shutil.rmtree(params['out_dir'])

def load_audio():
    path = Path('./tests/audio_examples/')
    audio_files = path.rglob('*.flac')
    audio_files = [a for a in audio_files]

    wav1, sr = torchaudio.load(audio_files[0])
    sample1 = {'waveform': wav1, 'sample_rate':sr, 'targets':[1.0,1.0,1.0,0.0]}
    
    wav2, sr = torchaudio.load(audio_files[1])
    sample2 = {'waveform': wav2, 'sample_rate':sr, 'targets':[0.0,0.0,0.0,1.0]}

    return sample1, sample2 

def test_pooling():
    sample1, _ = load_audio()
    params = {'out_dir':Path('./out_dir'), 'model_type':'wavlm-base', 'pool_method': 'max'}
    e = HFExtractor(model_type='wavlm-base')
    sample = e(sample1)
    features = sample['waveform']
    
    #MAX POOL
    m = HFModel(**params)
    output = m.base_model(features.to(m.device))
    output = output['last_hidden_state']

    pool_max = m.pooling(output).detach()
    assert pool_max.ndim == 2 and pool_max.shape[0] == 1 and pool_max.shape[1] == output.shape[-1], 'Max pooling not outputting proper shape'

    #MEAN POOL
    params['pool_method'] = 'mean'
    m = HFModel(**params)
    output = m.base_model(features.to(m.device))
    output = output['last_hidden_state']

    pool_mean = m.pooling(output)
    assert pool_mean.ndim == 2 and pool_mean.shape[0] == 1 and pool_mean.shape[1] == output.shape[-1], 'Mean pooling not outputting proper shape'

    #ATTENTION POOL
    params['pool_method'] = 'attention'
    m = HFModel(**params)
    output = m.base_model(features.to(m.device))
    output = output['last_hidden_state']

    pool_attn = m.pooling(output)
    assert pool_attn.ndim == 2 and pool_attn.shape[0] == 1 and pool_attn.shape[1] == output.shape[-1], 'Attention pooling not outputting proper shape'

@pytest.mark.hf
def test_forward():
    sample1, _ = load_audio()
    params = {'out_dir':Path('./out_dir'), 'model_type':'wavlm-base', 'pool_method': 'max'}

    #test WavLM
    wavlme = HFExtractor(model_type='wavlm-base')
    wavlmsample = wavlme(sample1)
    wavlmfeatures = wavlmsample['waveform']
    wavlm = HFModel(**params)
    output = wavlm(wavlmfeatures)
    assert output.shape[0] == 1 and output.shape[1] == 1, 'outputs correct output features'

    #test hubert
    params['model_type'] = 'hubert-base'
    huberte = HFExtractor(model_type='hubert-base')
    hubertsample = huberte(sample1)
    hubertfeatures = hubertsample['waveform']
    hubert = HFModel(**params)
    output = hubert(hubertfeatures)
    assert output.shape[0] == 1 and output.shape[1] == 1, 'outputs correct output features'

    #test whisper
    whispere = HFExtractor(model_type='whisper-tiny')
    whispersample = whispere(sample1)
    whisperfeatures = whispersample['waveform']
    params['model_type'] = 'whisper-tiny'
    whisper = HFModel(**params)
    output = whisper(whisperfeatures)
    assert output.shape[0] == 1 and output.shape[1] == 1, 'outputs correct output features'

