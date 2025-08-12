"""
Test hugging face models

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
import json
from pathlib import Path
import shutil

##third-party
from google.cloud import storage
import pytest
import torchaudio
import torch

##local
from summer25.models import HFModel
from summer25.io import search_gcs
from summer25.constants import _MODELS

##### HELPER FUNCTIONS#####
def load_json():
    with open('./private_loading/gcs.json', 'r') as file:
        data = json.load(file)

    gcs_prefix = data['gcs_prefix']
    checkpoint_prefix = data['checkpoint_prefix']
    bucket_name = data['bucket_name']
    project_name = data['project_name']
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.get_bucket(bucket_name)
    return gcs_prefix, checkpoint_prefix, bucket

def remove_gcs_directories(gcs_prefix,bucket, directory='test_split', pattern="*"):
    dir = gcs_prefix + f'{directory}'
    existing = search_gcs(pattern, dir, bucket)
    for e in existing:
        blob = bucket.blob(e)
        blob.delete()
    existing = search_gcs(pattern, dir, bucket)
    assert existing == []

def check_requires_grad(model):
    unfrozen = model.unfreeze
    check = []
    for name, param in model.base_model.named_parameters():
        if any([f in name for f in unfrozen]):
            check.append(param.requires_grad)
        else:
            check.append(param.requires_grad is False)
    return all(check)

def check_exclude_decoder(model):
    check = []
    for name, param in model.base_model.named_parameters():
        if any(['decoder' in name]):
            check.append(param.requires_grad is False)
    return all(check)

def check_lora(model):
    check = []
    for name, param in model.base_model.named_parameters():
        if 'lora' in name and 'encoder' in name and ('k_proj' in name or 'v_proj' in name or 'q_proj' in name):
            check.append(param.requires_grad)
        else:
            check.append(param.requires_grad is False)
    return all(check)

def check_softprompt(model):
    check = []
    for name, param in model.base_model.named_parameters():
        if 'prompt_encoder' in name:
            check.append(param.requires_grad)
            check.append(param.shape[0] == 4)
        else:
            check.append(param.requires_grad is False)
    return all(check)

def load_audio():
    path = Path('./tests/audio_examples/')
    audio_files = path.rglob('*.flac')
    audio_files = [a for a in audio_files]

    wav1, sr = torchaudio.load(audio_files[0])
    sample1 = {'waveform': wav1, 'sample_rate':sr, 'targets':[1.0,1.0,1.0,0.0]}
    
    wav2, sr = torchaudio.load(audio_files[1])
    sample2 = {'waveform': wav2, 'sample_rate':sr, 'targets':[0.0,0.0,0.0,1.0]}

    return sample1, sample2 

def mock_forward(m, waveform):
    inputs, attention_mask = m.feature_extractor(waveform)
    attention_mask = attention_mask.to(m.device)
    if m.is_whisper_model:
        output = m.base_model.encoder(inputs, attention_mask=attention_mask)
    else:
        output = m.base_model(inputs, attention_mask=attention_mask)
    output = output['last_hidden_state']

    ds_attn_mask = m._downsample_attention_mask(attn_mask=attention_mask.to(torch.float16), target_len=output.shape[1])
    expand_attn_mask = ds_attn_mask.unsqueeze(-1).repeat(1, 1, output.shape[2])
    output[~(expand_attn_mask==1.0)] = 0.0

    pool = m._pool(output, attn_mask=ds_attn_mask).detach()
    assert pool.ndim == 2 and pool.shape[0] == 2 and pool.shape[1] == output.shape[-1], f'{m.pool_method} pooling not outputting proper shape'

def mock_forward_noattn(m, waveform):
    inputs, attention_mask = m.feature_extractor(waveform)
    output = m.base_model(inputs.to(m.device))
    output = output['last_hidden_state']
    pool = m._pool(output).detach()
    assert pool.ndim == 2 and pool.shape[0] == 2 and pool.shape[1] == output.shape[-1], f'{m.pool_method} pooling not outputting proper shape'

##### TESTS #####
def test_hfmodel_init():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)
    assert m is not None, 'Model not running properly.'
    assert m.base_model is None, "Base model should be None"
    assert m.feature_extractor is None, "Feature extractor should be None"
    assert m.classifier_head is not None, "Clasifier head should not be None"

    #invalid model type (not hugging face)
    params['model_type']='test_model'
    with pytest.raises(AssertionError):
        m = HFModel(**params)

@pytest.mark.hf
def test_hfmodel_clf_io():
    params = {'out_dir':Path('./out_dir'), 'model_type':'wavlm-base'}
    m = HFModel(**params)
    
    #runs with checkpoint = None
    m.load_clf_checkpoint(checkpoint=None)

    #no bucket, checkpoint does not exist
    ckpt = Path('./ckpt')
    if ckpt.exists():
        shutil.rmtree(ckpt)

    with pytest.raises(AssertionError):
        m.load_clf_checkpoint(checkpoint=ckpt)

    #checkpoint an existing file but not a .pth/.pt file
    with pytest.raises(AssertionError):
        m.load_clf_checkpoint(checkpoint = params['out_dir'])
    
    with pytest.raises(AssertionError):
        m.load_clf_checkpoint(checkpoint = 'tests/audio_examples/1919-142785-0008.flac')

    #SAVE CHECKPOINT
    #give incorrect path
    with pytest.raises(AssertionError):
        m._save_clf_checkpoint(path=params['out_dir'])
    
    #give possible path
    out_path = params['out_dir'] / 'test_checkpoint.pt'
    m._save_clf_checkpoint(path=out_path)
    assert out_path.exists(), 'Incorrect saving'

    #LOAD EXISTING & COMPATIBLE CHECKPOINT
    m.load_clf_checkpoint(checkpoint=out_path)

    #TRY TO LOAD EXISTInG BUT INCOMPATIBLE CHECKPOINT
    params['clf_type'] = 'transformer'
    m2 = HFModel(**params)
    out_path2 = params['out_dir'] / 'test_checkpoint2.pt'
    m2._save_clf_checkpoint(path=out_path2)

    with pytest.raises(ValueError):
        m.load_clf_checkpoint(checkpoint=out_path2)

    shutil.rmtree(params['out_dir'])

@pytest.mark.gcs
def test_hfmodel_clf_io_gcs():
    gcs_prefix, ckpt_prefix, bucket = load_json()
    params = {'out_dir':f'{gcs_prefix}test_clf', 'bucket':bucket, 'model_type':'wavlm-base'}
    m = HFModel(**params)
    
    #runs with checkpoint = None
    m.load_clf_checkpoint(checkpoint=None)

    #bucket, checkpoint does not exist
    ckpt = params['out_dir']
    with pytest.raises(AssertionError):
        m.load_clf_checkpoint(checkpoint=ckpt)

    #checkpoint an existing file but not a .pth/.pt file
    ckpt = f'{gcs_prefix}test_audio/test1.wav'
    with pytest.raises(AssertionError):
        m.load_clf_checkpoint(checkpoint = params['out_dir'])
    with pytest.raises(AssertionError):
        m.load_clf_checkpoint(checkpoint = ckpt)
    
    #SAVE CHECKPOINT
    #give incorrect path
    with pytest.raises(AssertionError):
        m._save_clf_checkpoint(path=params['out_dir'])
    
    #give possible path
    p = params['out_dir']
    out_path = f'{p}/test_checkpoint.pt'
    m._save_clf_checkpoint(path=out_path)
    existing = search_gcs(pattern =out_path, directory=out_path, bucket=bucket)
    assert existing != [], 'Incorrect saving'
    temp_path = Path('.') / 'test_checkpoint.pt'
    assert not temp_path.exists(), 'Did not save correctly'            

    #LOAD EXISTING & COMPATIBLE CHECKPOINT
    m.load_clf_checkpoint(checkpoint=out_path)
    assert temp_path.exists(), 'Deleted checkpoint without delete_download set to True'

    m.load_clf_checkpoint(checkpoint=out_path, delete_download=True)
    assert not temp_path.exists(), 'Did not delete checkpoint when specified.' 

    #TRY TO LOAD EXISTInG BUT INCOMPATIBLE CHECKPOINT
    params['clf_type'] = 'transformer'
    m2 = HFModel(**params)
    out_path2 = f'{p}/test_checkpoint2.pt'
    m2._save_clf_checkpoint(path=out_path2)
    existing = search_gcs(pattern =out_path2, directory=out_path2, bucket=bucket)
    assert existing != [], 'Incorrect saving'
    temp_path2 = Path('.') / 'test_checkpoint2.pt'
    assert not temp_path2.exists(), 'Did not save correctly'            

    with pytest.raises(ValueError):
        m.load_clf_checkpoint(checkpoint=out_path2)
    
    assert temp_path2.exists(), 'Deleted download without command'

    with pytest.raises(ValueError):
        m.load_clf_checkpoint(checkpoint=out_path2, delete_download=True)
    assert not temp_path2.exists(), 'Did not delete correctly.'

    remove_gcs_directories(gcs_prefix, bucket, 'test_clf')
    existing = search_gcs(params['out_dir'], params['out_dir'], bucket)
    assert existing == [], 'Not all checkpoints deleted'     

@pytest.mark.hf
def test_hfmodel_pretrained():
    params = {'out_dir':Path('./out_dir'), 'model_type':'wavlm-base'}

    #from hub
    m = HFModel(**params)
    m.load_model_checkpoint(checkpoint=m.hf_hub, from_hub=True)
    assert m.base_model is not None 
    assert m.local_path is None

    #from hub, test_hub_fail, delete_download = False
    m2 = HFModel(**params)
    m2.load_model_checkpoint(checkpoint=m2.hf_hub, from_hub=True, test_hub_fail=True)
    assert m2.base_model is not None 
    assert m2.local_path is not None
    assert m2.local_path.exists()


    #from hub, test_local_fail
    m3 = HFModel(**params)
    with pytest.raises(AssertionError):
        m3.load_model_checkpoint(checkpoint=Path('./random'), from_hub=True, test_local_fail=True)
    with pytest.raises(AssertionError):
        m3.load_model_checkpoint(checkpoint=Path('tests/audio_examples/1919-142785-0008.flac'), from_hub=True, test_local_fail=True)
    m3.load_model_checkpoint(checkpoint=m2.local_path, from_hub=True, test_local_fail=True)
    assert m3.base_model is not None
    assert m3.local_path == m2.local_path

    #from hub is false
    m4 = HFModel(**params)
    with pytest.raises(AssertionError):
        m4.load_model_checkpoint(checkpoint=Path('./random'), from_hub=False)
    with pytest.raises(AssertionError):
        m4.load_model_checkpoint(checkpoint=Path('tests/audio_examples/1919-142785-0008.flac'), from_hub=False)
    m4.load_model_checkpoint(checkpoint=m2.local_path, from_hub=False)
    assert m4.base_model is not None
    assert m4.local_path == m2.local_path

    #unexpected model type
    m5 = HFModel(**params)
    with pytest.raises(ValueError):
        m5.load_model_checkpoint(checkpoint=_MODELS['whisper-base']['hf_hub'], from_hub=True)

    #from hub, delete_download = True
    m6 = HFModel(**params)
    m6.load_model_checkpoint(checkpoint=m6.hf_hub, from_hub=True, test_hub_fail=True, delete_download=True)
    assert m6.base_model is not None 
    assert m6.local_path is not None
    assert not m6.local_path.exists()

    shutil.rmtree(params['out_dir'])

@pytest.mark.gcs
def test_hfmodel_pretrained_gcs():
    gcs_prefix, ckpt_prefix, bucket = load_json()
    params = {'out_dir':f'{gcs_prefix}test_model', 'bucket':bucket, 'model_type':'wavlm-base'}

    #from hub, test_local_fail
    m = HFModel(**params)
    with pytest.raises(AssertionError):
        m.load_model_checkpoint(checkpoint=f'{gcs_prefix}random', from_hub=True, test_local_fail=True)
    with pytest.raises(AssertionError):
        m.load_model_checkpoint(checkpoint=f'{gcs_prefix}test_audio/test1.wav', from_hub=True, test_local_fail=True)

    m.load_model_checkpoint(checkpoint=ckpt_prefix, from_hub=True, test_local_fail=True)
    assert m.base_model is not None
    assert m.local_path.exists()

    #from hub is false
    m2 = HFModel(**params)
    with pytest.raises(AssertionError):
        m2.load_model_checkpoint(checkpoint=f'{gcs_prefix}random', from_hub=True, test_local_fail=True)
    with pytest.raises(AssertionError):
        m2.load_model_checkpoint(checkpoint=f'{gcs_prefix}test_audio/test1.wav', from_hub=True, test_local_fail=True)

    m2.load_model_checkpoint(checkpoint=ckpt_prefix, from_hub=False)
    assert m2.base_model is not None
    assert m2.local_path == m.local_path
    assert m2.local_path.exists()

    #unexpected model type
    params['model_type'] = 'whisper-base'
    m3 = HFModel(**params)
    with pytest.raises(ValueError):
        m3.load_model_checkpoint(checkpoint=ckpt_prefix, from_hub=False)

    #delete download
    params['model_type'] = 'wavlm-base'
    m4 = HFModel(**params)
    m4.load_model_checkpoint(checkpoint=ckpt_prefix,from_hub=False, delete_download=True)
    assert m4.base_model is not None 
    assert m4.local_path is not None
    assert not m4.local_path.exists()

    remove_gcs_directories(gcs_prefix, bucket, 'test_model')
    existing = search_gcs(params['out_dir'], params['out_dir'], bucket)
    assert existing == [], 'Not all checkpoints deleted'

@pytest.mark.hf
def test_hfmodel_finetuned():
    params = {'out_dir':Path('./out_dir'), 'model_type':'wavlm-base'}

    #base
    m = HFModel(**params)
    m.load_model_checkpoint(checkpoint=m.hf_hub, from_hub=True)
    assert m.base_model is not None 
    assert m.local_path is None
    
    #save model
    p = params['out_dir']/'test_model'
    m._save_model_checkpoint(path=p)
    assert p.exists(), 'Model checkpoint not saved correctly'

    # create new model with finetuned model path
    m2 = HFModel(**params)
    m2.load_model_checkpoint(checkpoint=p, from_hub=False)
    assert m2.base_model is not None
    assert m2.local_path == p

    shutil.rmtree(params['out_dir'])


@pytest.mark.gcs
def test_hfmodel_finetuned_gcs():
    gcs_prefix, ckpt_prefix, bucket = load_json()
    params = {'out_dir':f'{gcs_prefix}test_model', 'bucket':bucket, 'model_type':'wavlm-base'}

    #base
    m = HFModel(**params)
    m.load_model_checkpoint(checkpoint=m.hf_hub, from_hub=True)
    assert m.base_model is not None 
    assert m.local_path is None
    
    #save model
    m._save_model_checkpoint(path=params['out_dir'])
    assert not Path('./test_model').exists(), 'Model checkpoint not saved correctly'
    existing = search_gcs(params['out_dir'], params['out_dir'], bucket)
    assert existing != [], 'Model checkpoint not saved correctly'

    # create new model with finetuned model path
    m2 = HFModel(**params)
    m2.load_model_checkpoint(checkpoint=params['out_dir'], from_hub=False, delete_download=True)
    assert m2.base_model is not None
    

    remove_gcs_directories(gcs_prefix, bucket, 'test_model')
    existing = search_gcs(params['out_dir'], params['out_dir'], bucket)
    assert existing == [], 'Not all checkpoints deleted'   

@pytest.mark.hf
def test_lora():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    params['finetune_method'] = 'lora'
    m = HFModel(**params)

    m.load_model_checkpoint(checkpoint=m.hf_hub, from_hub=True)
    m.configure_peft(checkpoint=m.hf_hub, checkpoint_type='pt')
    assert check_lora(m)
    n_params = sum(p.numel() for p in m.base_model.parameters() if p.requires_grad)

    #save
    p = params['out_dir']/'test_model'
    m._save_model_checkpoint(path=p)
    assert p.exists(), 'Model checkpoint not saved correctly'

    # LOAD FINETUNED
    m2 = HFModel(**params)
    m2.load_model_checkpoint(checkpoint=p)
    m2.configure_peft(checkpoint=p, checkpoint_type='ft')
    assert check_lora(m2)
    n_params2 = sum(p.numel() for p in m2.base_model.parameters() if p.requires_grad)
    assert n_params == n_params2, 'Different models during lora'
    
    # TEST WHISPER
    params['model_type'] = 'whisper-tiny'
    params['from_hub'] = True
    m3 = HFModel(**params)
    m3.load_model_checkpoint(checkpoint=m3.hf_hub)
    assert check_lora(m3)
    assert check_exclude_decoder(m3)
    
    shutil.rmtree(params['out_dir'])

@pytest.mark.gcs
def test_lora_gcs():
    gcs_prefix, ckpt_prefix, bucket = load_json()
    params = {'out_dir':f'{gcs_prefix}test_model', 'bucket':bucket, 'model_type':'wavlm-base', 'finetune_method':'lora'}
    m = HFModel(**params)

    m.load_model_checkpoint(checkpoint=ckpt_prefix, from_hub=False)
    m.configure_peft(checkpoint=ckpt_prefix, checkpoint_type='pt', delete_download = True)
    assert check_lora(m)
    n_params = sum(p.numel() for p in m.base_model.parameters() if p.requires_grad)

    #save
    m._save_model_checkpoint(path=params['out_dir'])
    assert not Path('./test_model').exists(), 'Model checkpoint not saved correctly'
    existing = search_gcs(params['out_dir'], params['out_dir'], bucket)
    assert existing != [], 'Model checkpoint not saved correctly'

    m2 = HFModel(**params)
    m2.load_model_checkpoint(checkpoint=params['out_dir'], from_hub=False)
    m2.configure_peft(checkpoint=params['out_dir'], checkpoint_type='ft', delete_download=True)
    assert check_lora(m2)
    n_params2 = sum(p.numel() for p in m2.base_model.parameters() if p.requires_grad)
    assert n_params == n_params2, 'Different models during lora'

    remove_gcs_directories(gcs_prefix, bucket, 'test_model')
    existing = search_gcs(params['out_dir'], params['out_dir'], bucket)
    assert existing == [], 'Not all checkpoints deleted'

@pytest.mark.hf
def test_softprompt():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    params['finetune_method'] = 'soft-prompt'
    m = HFModel(**params)

    m.load_model_checkpoint(checkpoint=m.hf_hub, from_hub=True)
    m.configure_peft(checkpoint=m.hf_hub, checkpoint_type='pt')
    assert check_softprompt(m)
    n_params = sum(p.numel() for p in m.base_model.parameters() if p.requires_grad)

    #save
    p = params['out_dir']/'test_model'
    m._save_model_checkpoint(path=p)
    assert p.exists(), 'Model checkpoint not saved correctly'

    # LOAD FINETUNED
    m2 = HFModel(**params)
    m2.load_model_checkpoint(checkpoint=m.hf_hub)
    m2.configure_peft(checkpoint=p, checkpoint_type='ft')
    assert check_softprompt(m2)
    n_params2 = sum(p.numel() for p in m2.base_model.parameters() if p.requires_grad)
    assert n_params == n_params2, 'Different models during lora'
    
    # TEST WHISPER
    params['model_type'] = 'whisper-tiny'
    params['from_hub'] = True
    m3 = HFModel(**params)
    m3.load_model_checkpoint(checkpoint=m3.hf_hub)
    assert check_softprompt(m3)
    assert check_exclude_decoder(m3)
    
    shutil.rmtree(params['out_dir'])

@pytest.mark.gcs
def test_softprompt_gcs():
    gcs_prefix, ckpt_prefix, bucket = load_json()
    params = {'out_dir':f'{gcs_prefix}test_model', 'bucket':bucket, 'model_type':'wavlm-base', 'finetune_method':'soft-prompt'}
    m = HFModel(**params)

    m.load_model_checkpoint(checkpoint=ckpt_prefix, from_hub=False)
    m.configure_peft(checkpoint=ckpt_prefix, checkpoint_type='pt', delete_download = True)
    assert check_softprompt(m)
    n_params = sum(p.numel() for p in m.base_model.parameters() if p.requires_grad)

    #save
    m._save_model_checkpoint(path=params['out_dir'])
    assert not Path('./test_model').exists(), 'Model checkpoint not saved correctly'
    existing = search_gcs(params['out_dir'], params['out_dir'], bucket)
    assert existing != [], 'Model checkpoint not saved correctly'

    m2 = HFModel(**params)
    m2.load_model_checkpoint(checkpoint=ckpt_prefix, from_hub=False)
    m2.configure_peft(checkpoint=params['out_dir'], checkpoint_type='ft', delete_download=True)
    assert check_softprompt(m2)
    n_params2 = sum(p.numel() for p in m2.base_model.parameters() if p.requires_grad)
    assert n_params == n_params2, 'Different models during lora'

    remove_gcs_directories(gcs_prefix, bucket, 'test_model')
    existing = search_gcs(params['out_dir'], params['out_dir'], bucket)
    assert existing == [], 'Not all checkpoints deleted'

def test_freeze():
    params = {'out_dir':Path('./out_dir'), 'model_type':'wavlm-base'}

    #freeze all
    #no freezing (required only)
    params['freeze_method'] = 'all'
    m = HFModel(**params)
    m.load_model_checkpoint(checkpoint=m.hf_hub)
    assert m is not None, 'Model not running properly.'
    assert check_requires_grad(m)

    #no freezing (required only)
    params['freeze_method'] = 'required-only'
    m0 = HFModel(**params)
    m0.load_model_checkpoint(checkpoint=m0.hf_hub)
    assert m is not None, 'Model not running properly.'
    assert check_requires_grad(m0)

    #freeze optional
    params['freeze_method'] = 'optional'
    m1 = HFModel(**params)
    m1.load_model_checkpoint(checkpoint=m1.hf_hub)
    assert m1 is not None, 'Model not running properly.'
    assert check_requires_grad(m1)

    #freeze half
    params['freeze_method'] = 'half'
    m2 = HFModel(**params)
    m2.load_model_checkpoint(checkpoint=m2.hf_hub)
    assert m2 is not None, 'Model not running properly.'
    assert check_requires_grad(m2)
    assert len(m2.unfreeze) == 6

    #freeze all but last
    params['freeze_method'] = 'exclude-last'
    m3 = HFModel(**params)
    m3.load_model_checkpoint(checkpoint=m3.hf_hub)
    assert m3 is not None, 'Model not running properly.'
    assert check_requires_grad(m3)
    assert len(m3.unfreeze) == 1

    #freeze specific layers
    params['freeze_method'] = 'layer'
    params['unfreeze_layers'] = ['encoder.layers.5']
    m4 = HFModel(**params)
    m4.load_model_checkpoint(checkpoint=m4.hf_hub)
    assert m4 is not None, 'Model not running properly.'
    assert check_requires_grad(m4)
    assert len(m4.unfreeze) == 1 and m4.unfreeze == params['unfreeze_layers']

    #TRY WITH HUBERT
    params['model_type'] = 'hubert-base'
    params['freeze_method'] = 'optional'
    del params['unfreeze_layers']
    m5 = HFModel(**params)
    m5.load_model_checkpoint(checkpoint=m5.hf_hub)
    assert m5 is not None

    #TRY WITH WHISPER
    params['model_type'] = 'whisper-tiny'
    params['freeze_method'] = 'optional'
    m6 = HFModel(**params)
    m6.load_model_checkpoint(checkpoint=m6.hf_hub)
    assert m6 is not None, 'Model not running properly.'
    assert check_requires_grad(m6)
    assert check_exclude_decoder(m6)

    #Check whisper with half and exclude last 
    params['freeze_method'] = 'half'
    m7 = HFModel(**params)
    m7.load_model_checkpoint(checkpoint=m7.hf_hub)
    assert m7 is not None, 'Model not running properly.'
    assert check_requires_grad(m7)
    assert len(m7.unfreeze) == 3 
    assert check_exclude_decoder(m7)

    params['freeze_method'] = 'exclude-last'
    m8 = HFModel(**params)
    m8.load_model_checkpoint(checkpoint=m8.hf_hub)
    assert m8 is not None, 'Model not running properly.'
    assert check_requires_grad(m8)
    assert len(m8.unfreeze) == 2
    assert check_exclude_decoder(m7)

    shutil.rmtree(params['out_dir'])

def test_pooling():
    sample1, sample2 = load_audio()
    waveforms = [sample1['waveform'], sample2['waveform']]
    params = {'out_dir':Path('./out_dir'), 'model_type':'wavlm-base', 'pool_method': 'max'}

    #MAX POOL
    m = HFModel(**params)
    m.load_model_checkpoint(checkpoint=m.hf_hub)
    m.load_feature_extractor(checkpoint=m.hf_hub)
    mock_forward(m, waveforms)

    #MEAN POOL
    params['pool_method'] = 'mean'
    m = HFModel(**params)
    m.load_model_checkpoint(checkpoint=m.hf_hub)
    m.load_feature_extractor(checkpoint=m.hf_hub)
    mock_forward(m, waveforms)
    ### POOL WITHOUT ATTN MASK
    mock_forward_noattn(m, waveforms)

    #ATTENTION POOL
    params['pool_method'] = 'attention'
    m = HFModel(**params)
    m.load_model_checkpoint(checkpoint=m.hf_hub)
    m.load_feature_extractor(checkpoint=m.hf_hub)
    mock_forward(m, waveforms)
    ### POOL WITHOUT ATTN MASK
    mock_forward_noattn(m, waveforms)

    #WHISPER POOL
    params['model_type'] = 'whisper-tiny'
    params['pool_method'] = 'mean'
    m = HFModel(**params)
    m.load_model_checkpoint(checkpoint=m.hf_hub)
    m.load_feature_extractor(checkpoint=m.hf_hub)
    mock_forward(m, waveforms)

    shutil.rmtree(params['out_dir'])

@pytest.mark.hf
def test_forward():
    sample1, _ = load_audio()
    waveforms = [sample1['waveform']]
    params = {'out_dir':Path('./out_dir'), 'model_type':'wavlm-base', 'pool_method': 'max'}

    #test WavLM
    wavlm = HFModel(**params)
    wavlm.load_model_checkpoint(checkpoint=wavlm.hf_hub)
    wavlm.load_feature_extractor(checkpoint=wavlm.hf_hub)
    output = wavlm(waveforms)
    assert output.shape[0] == 1 and output.shape[1] == 1, 'outputs correct output features'

    #test hubert
    params['model_type'] = 'hubert-base'
    hubert = HFModel(**params)
    hubert.load_model_checkpoint(checkpoint=hubert.hf_hub)
    hubert.load_feature_extractor(checkpoint=hubert.hf_hub)
    output = hubert(waveforms)
    assert output.shape[0] == 1 and output.shape[1] == 1, 'outputs correct output features'

    #test whisper
    params['model_type'] = 'whisper-tiny'
    whisper = HFModel(**params)
    whisper.load_model_checkpoint(checkpoint=whisper.hf_hub)
    whisper.load_feature_extractor(checkpoint=whisper.hf_hub)
    output = whisper(waveforms)
    assert output.shape[0] == 1 and output.shape[1] == 1, 'outputs correct output features'
    
    shutil.rmtree(params['out_dir'])

@pytest.mark.gcs
def test_forward_gcs():
    gcs_prefix, ckpt_prefix, bucket = load_json()
    params = {'model_type': 'wavlm-base', 'out_dir':f'{gcs_prefix}test_model', 'bucket':bucket}
    sample1, sample2 = load_audio()
    waveforms = [sample1['waveform'], sample2['waveform']]

    m = HFModel(**params)
    m.load_model_checkpoint(checkpoint=ckpt_prefix, from_hub=False)
    m.load_feature_extractor(checkpoint=ckpt_prefix, from_hub=False)
    assert m is not None, 'Model not running properly.'
    output = m(waveforms)
    assert output.shape[0] == 2 and output.shape[1] == 1, 'outputs correct output features'
    
    remove_gcs_directories(gcs_prefix, bucket, 'test_model')
    existing = search_gcs(params['out_dir'], params['out_dir'], bucket)
    assert existing == [], 'Not all checkpoints deleted'

@pytest.mark.hf
def test_peft_forward():
    sample1, sample2 = load_audio()
    waveforms = [sample1['waveform'], sample2['waveform']]
    #wavlm - lora
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    params['finetune_method'] = 'lora'
    m = HFModel(**params)
    m.load_model_checkpoint(checkpoint=m.hf_hub)
    m.load_feature_extractor(checkpoint=m.hf_hub)
    m.configure_peft(checkpoint=m.hf_hub)
    n_params = sum(p.numel() for p in m.base_model.parameters() if p.requires_grad)
    output1 = m(waveforms)
    assert output1.shape[0] == 2 and output1.shape[1] == 1, 'outputs correct output features'
    #wavlm - lora, reloaded
    m._save_model_checkpoint(path = params['out_dir']/'lora')

    m2 = HFModel(**params)
    m2.load_model_checkpoint(checkpoint=m2.hf_hub, from_hub=False)
    m2.load_feature_extractor(checkpoint=m2.hf_hub, from_hub=True)
    m2.configure_peft(checkpoint=params['out_dir'] /'lora', checkpoint_type='ft')
    n_params2 = sum(p.numel() for p in m2.base_model.parameters() if p.requires_grad)
    assert n_params == n_params2, 'Different models during lora'
    output2 = m2(waveforms)
    assert torch.equal(output1, output2)

    shutil.rmtree(params['out_dir'])

    #wavlm - soft prompt
    params['finetune_method'] = 'soft-prompt'
    m = HFModel(**params)
    m.load_model_checkpoint(checkpoint=m.hf_hub)
    m.load_feature_extractor(checkpoint=m.hf_hub)
    m.configure_peft(checkpoint=m.hf_hub)
    n_params = sum(p.numel() for p in m.base_model.parameters() if p.requires_grad)
    output1 = m(waveforms)
    assert output1.shape[0] == 2 and output1.shape[1] == 1, 'outputs correct output features'
    #wavlm - lora, reloaded
    m._save_model_checkpoint(path = params['out_dir'] / 'softprompt')

    m2 = HFModel(**params)
    m2.load_model_checkpoint(checkpoint=m2.hf_hub, from_hub=False)
    m2.load_feature_extractor(checkpoint=m2.hf_hub, from_hub=True)
    m2.configure_peft(checkpoint=params['out_dir'] /'softprompt', checkpoint_type='ft')
    n_params2 = sum(p.numel() for p in m2.base_model.parameters() if p.requires_grad)
    assert n_params == n_params2, 'Different models during soft-prompt'
    output2 = m2(waveforms)
    assert torch.equal(output1, output2)

    shutil.rmtree(params['out_dir'])

@pytest.mark.gcs
def test_peft_forward_lora_gcs():
    gcs_prefix, ckpt_prefix, bucket = load_json()
    params = {'model_type': 'wavlm-base', 'out_dir':Path(f'{gcs_prefix}test_model'), 'bucket':bucket}
    sample1, sample2 = load_audio()
    waveforms = [sample1['waveform'], sample2['waveform']]

    #base test
    params['model_type'] = 'wavlm-base'
    params['finetune_method'] = 'lora'
    m = HFModel(**params)
    m.load_model_checkpoint(checkpoint=ckpt_prefix, from_hub=False)
    m.load_feature_extractor(checkpoint=ckpt_prefix, from_hub=False)
    m.configure_peft(checkpoint=ckpt_prefix, from_hub=False, delete_download=True)
    n_params = sum(p.numel() for p in m.base_model.parameters() if p.requires_grad)
    output1 = m(waveforms)
    assert output1.shape[0] == 2 and output1.shape[1] == 1, 'outputs correct output features'
    #wavlm - lora, reloaded
    m._save_model_checkpoint(path = params['out_dir']/'lora')

    m2 = HFModel(**params)
    m2.load_model_checkpoint(checkpoint=ckpt_prefix, from_hub=False)
    m2.load_feature_extractor(checkpoint=ckpt_prefix, from_hub=False)
    m2.configure_peft(checkpoint=params['out_dir'] /'lora', checkpoint_type='ft')
    n_params2 = sum(p.numel() for p in m2.base_model.parameters() if p.requires_grad)
    assert n_params == n_params2, 'Different models during lora'
    output2 = m2(waveforms)
    assert torch.equal(output1, output2)

    remove_gcs_directories(gcs_prefix, bucket, 'test_model')

@pytest.mark.gcs
def test_peft_forward_softprompt_gcs():
    gcs_prefix, ckpt_prefix, bucket = load_json()
    params = {'model_type': 'wavlm-base', 'out_dir':Path(f'{gcs_prefix}test_model'), 'bucket':bucket}
    sample1, sample2 = load_audio()
    waveforms = [sample1['waveform'], sample2['waveform']]

    #wavlm - soft prompt
    params['finetune_method'] = 'soft-prompt'
    m3 = HFModel(**params)
    m3.load_model_checkpoint(checkpoint=ckpt_prefix, from_hub=False)
    m3.load_feature_extractor(checkpoint=ckpt_prefix, from_hub=False)
    m3.configure_peft(checkpoint=ckpt_prefix, from_hub=False)
    n_params = sum(p.numel() for p in m3.base_model.parameters() if p.requires_grad)
    output1 = m3(waveforms)
    assert output1.shape[0] == 2 and output1.shape[1] == 1, 'outputs correct output features'
    #wavlm - lora, reloaded
    m3._save_model_checkpoint(path = params['out_dir'] / 'softprompt')

    m4 = HFModel(**params)
    m4.load_model_checkpoint(checkpoint=ckpt_prefix, from_hub=False)
    m4.load_feature_extractor(checkpoint=ckpt_prefix, from_hub=False, delete_download=False)
    m4.configure_peft(checkpoint=params['out_dir'] /'softprompt', checkpoint_type='ft')
    n_params2 = sum(p.numel() for p in m4.base_model.parameters() if p.requires_grad)
    assert n_params == n_params2, 'Different models during soft-prompt'
    output2 = m4(waveforms)
    assert torch.equal(output1, output2)

    remove_gcs_directories(gcs_prefix, bucket, 'test_model')