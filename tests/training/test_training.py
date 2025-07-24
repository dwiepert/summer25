"""
Test trainer

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
import os
from pathlib import Path
import shutil
import json
##third-party
import pandas as pd
import pytest
from torch.utils.data import DataLoader

##local
from summer25.models import HFModel, HFExtractor
from summer25.dataset import WavDataset, collate_features
from summer25.training import Trainer
from summer25.constants import _FEATURES
from summer25.io import search_gcs

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

def check_checkpoints(m, name_prefix, checkpoints=[0], epochs=7, val=True, args=[], train_params={}, early_stop=False):
    if m.bucket:
        path = f'{m.out_dir}{name_prefix}'
        existing = search_gcs(path, path, m.bucket)
        assert any(['train_log.json' in e for e in existing]), 'Training log not dumped correctly'
        out_path = f'{path}/weights'
        for c in checkpoints:
            name = f'checkpoint{c}_{m.model_name}'
            assert any([name in e for e in existing]), 'Model checkpoint not saved correctly'
        
        if not early_stop:
            name = f'final{epochs-1}_{m.model_name}'
            assert any([name in e for e in existing]), 'Final model not saved correctly'
        else:
            name = f'best{epochs-1}_{m.model_name}'
            assert any([name in e for e in existing]), 'Best model not saved correctly.'
        
        p = f'{path}/train_log.json'
        blob = m.bucket.blob(p)
        data = json.loads(blob.download_as_string())

    else:   
        path = m.out_dir / name_prefix 
        assert (path / 'train_log.json').exists(), 'Training log not dumped correctly'
        out_path = path / 'weights'
    
        for c in checkpoints:
            assert (out_path / f'checkpoint{c}_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    
        if not early_stop:
            assert (out_path / f'final{epochs-1}_{m.model_name}').exists(), 'Final model not saved correctly'
        else:
            assert(out_path / f'best{epochs-1}_{m.model_name}').exists(), 'Best model not saved correctly.'
        with open(str(path / 'train_log.json'), 'r') as f:
            data = json.load(f)

    assert len(data['train_loss']) == epochs
    if val:
        assert len(data['val_loss']) == epochs
    
    for a in args:
        assert a in data
        assert train_params[a] == data[a], 'Value not saved_correctly'

def data_dictionary():
    sub_list = ['1919-142785-0008']
    aud_list = ['1919-142785-0008']
    date_list = ['2025-01-01']
    task_name = ['sentence_repetition']
    feat1_list = [1.0]
    feat2_list = [1.0]

    data_dict = {'subject': sub_list, 
                'task_name': task_name,
                'incident_date': date_list, 
                'original_audio_id': aud_list, 
                _FEATURES[0]: feat1_list,
                _FEATURES[2]: feat2_list}

    data_df = pd.DataFrame(data_dict)
    data_df = data_df.set_index('subject')
    return data_df

def create_config(use_librosa=False, augment=False, truncate=False):
    config = {
        'use_librosa': use_librosa,
    }

    if truncate:
        config['truncate'] = {'length':5}
    if use_librosa:
        config['trim_level'] = 60
    else:
        config['trim_level'] = 7.0
    
    if augment:
         config['gauss'] = {'min_amplitude':0.001, 'max_amplitude':0.015}
    return config

def test_trainer_params():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)

    trainer_params = {'model': m, 'target_features':[_FEATURES[0]], 'early_stop':False}
    #initialize with valid params 
    t = Trainer(**trainer_params)
    assert not t.early_stop 

    #try invalid params
    trainer_params['optim_type'] = 'random'
    with pytest.raises(NotImplementedError):
        t = Trainer(**trainer_params)
    
    trainer_params['optim_type'] = 'adamw'
    trainer_params['loss_type'] = 'random'
    with pytest.raises(NotImplementedError):
        t = Trainer(**trainer_params)

    trainer_params['loss_type'] = 'bce'
    trainer_params['scheduler_type'] = 'random'
    with pytest.raises(NotImplementedError):
        t = Trainer(**trainer_params)

    #test different loss types implemented with valid params
    trainer_params['scheduler_type'] = None
    trainer_params['loss_type'] = 'rank'
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['rating_threshold'] = 2.0
    t = Trainer(**trainer_params)

    ##exponential
    trainer_params['scheduler_type'] = 'exponential'
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['end_lr'] = 0.001
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['epochs'] = 1
    t = Trainer(**trainer_params)
    trainer_params['gamma'] = 0.1
    t = Trainer(**trainer_params)

    ##warmup cosine
    trainer_params['scheduler_type'] = 'warmup-cosine'
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['warmup_epochs'] = 1 
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['train_len'] = 10
    t = Trainer(**trainer_params)

    ## cosine
    trainer_params['scheduler_type'] = 'cosine' 
    t = Trainer(**trainer_params)

    ## early stop
    del trainer_params['scheduler_type']
    trainer_params['early_stop'] = True
    t = Trainer(**trainer_params)

def test_fit():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)

    trainer_params = {'model': m, 'target_features':[_FEATURES[0]], 'early_stop':False, 'loss_type':'bce', 'scheduler_type':None, 'optim_type':'adamw', 'learning_rate':0.001, 'tf_learning_rate':0.0001}
    #initialize with valid params 
    t = Trainer(**trainer_params)
    
    df = data_dictionary()
    config = create_config()

    # AUDIO ONLY
    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config, target_labels=[_FEATURES[0]], bucket=None, extension='flac')
    
    #train_loader
    train_loader = DataLoader(d, batch_size=1, shuffle=True, collate_fn=collate_features)
    #val_loader
    val_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_features)

    #fit without val loader - check that train log is in output, run for 7 epochs, assert there is a checkpoint for 0,5 and final for e = 6
    t.fit(train_loader=train_loader, epochs=7)
    args = ['loss_type', 'optim_type', 'scheduler_type', 'learning_rate', 'tf_learning_rate']
    check_checkpoints(m,name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=False, args=args, train_params = trainer_params)

    #fit with val loader - check that train log is in output, run for 7 epochs, assert there is a checkpoint for 0,5 and final for e = 6
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m,name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True, args=args, train_params = trainer_params)
    shutil.rmtree(params['out_dir'])

def test_earlystop():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)

    trainer_params = {'model': m, 'target_features':[_FEATURES[0]], 'early_stop':True, 'loss_type': 'bce', 'optim_type':'adamw','learning_rate':0.0001, 'tf_learning_rate': 0.00001, 'scheduler_type':None}
    #fit with early stopping
    trainer_params['test'] = True
    t = Trainer(**trainer_params)

    df = data_dictionary()
    config = create_config()
    # AUDIO ONLY
    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config, target_labels=[_FEATURES[0]], bucket=None, extension='flac')
    #train_loader
    train_loader = DataLoader(d, batch_size=21, shuffle=True, collate_fn=collate_features)
    #val_loader
    val_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_features)

    args = ['loss_type', 'optim_type', 'scheduler_type', 'learning_rate', 'tf_learning_rate']
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m, name_prefix = f'{t.name_prefix}_e7', checkpoints=[], epochs = 1, val=True, args=args, train_params=trainer_params, early_stop=True)
    shutil.rmtree(params['out_dir'])

def test_schedulers():
    params = {'out_dir':Path('./out_dir')}
    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)

    trainer_params = {'model': m, 'target_features':[_FEATURES[0]], 'early_stop':False, 'loss_type': 'bce', 'optim_type':'adamw','learning_rate':0.0001, 'tf_learning_rate': 0.00001, 'scheduler_type':None}
    #initialize with valid params 
    df = data_dictionary()
    config = create_config()

    # AUDIO ONLY
    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config, target_labels=[_FEATURES[0]], bucket=None, extension='flac')

    #train_loader
    train_loader = DataLoader(d, batch_size=1, shuffle=True, collate_fn=collate_features)
    #val_loader
    val_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_features)

    args = ['loss_type', 'optim_type', 'scheduler_type', 'learning_rate', 'tf_learning_rate']
    #no scheduler 
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m,name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True, args=args, train_params = trainer_params)

    #exponential
    trainer_params['scheduler_type'] = 'exponential'
    #USE GAMMA
    #one gamma
    gamma = 0.01
    trainer_params['gamma'] = gamma
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m, name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True, args = args + ['gamma'], train_params = trainer_params)

    #tf gamma
    trainer_params['tf_gamma'] = 0.001
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m, name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True,  args = args + ['gamma', 'tf_gamma'], train_params = trainer_params)

    #NO GAMMA
    del trainer_params['gamma']
    del trainer_params['tf_gamma']
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['end_lr'] = 0.001
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['epochs'] = 7
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m, name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True,  args = args + ['end_lr', 'epochs'], train_params = trainer_params)

    #tf no gamma
    trainer_params['end_tf_lr'] = 0.01
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m,name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True, args = args + ['end_lr', 'epochs', 'end_tf_lr'], train_params = trainer_params)

    #cosine
    trainer_params['scheduler_type'] = 'cosine'
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['train_len'] = len(train_loader)
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m,name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True, args = args + ['train_len'], train_params = trainer_params)

    #warmup-cosine
    trainer_params['scheduler_type'] = 'warmup-cosine'
    del trainer_params['train_len']
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['train_len'] = len(train_loader)
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['warmup_epochs'] = 2
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m,name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True,  args = args + ['train_len', 'warmup_epochs'], train_params = trainer_params)

    trainer_params['tf_warmup_epochs'] = 3
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m, name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True, args = args + ['train_len', 'tf_warmup_epochs'], train_params = trainer_params)

    #test all assertions
    shutil.rmtree(params['out_dir'])

def test_loss():
    params = {'out_dir':Path('./out_dir')}
    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)

    trainer_params = {'model': m, 'target_features':[_FEATURES[0]], 'early_stop':False, 'loss_type': 'bce', 'optim_type':'adamw','learning_rate':0.0001, 'tf_learning_rate': 0.00001, 'scheduler_type':None}
    #initialize with valid params 
    df = data_dictionary()
    config = create_config()

    # AUDIO ONLY
    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config, target_labels=[_FEATURES[0]], bucket=None, extension='flac')

    #train_loader
    train_loader = DataLoader(d, batch_size=1, shuffle=True, collate_fn=collate_features)
    #val_loader
    val_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_features)

    args = ['loss_type', 'optim_type', 'scheduler_type', 'learning_rate', 'tf_learning_rate']
    #bce loss
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m,name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True, args=args, train_params = trainer_params)

    #rank loss
    trainer_params['loss_type'] = 'rank'
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['rating_threshold'] = 2.0
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m,name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True, args=args + ['rating_threshold'], train_params = trainer_params)

    #optional margin
    trainer_params['margin'] = 1.0
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m,name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True, args=args + ['rating_threshold', 'margin'], train_params = trainer_params)

    #option weight
    trainer_params['bce_weight'] = 0.25
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m,name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True, args=args + ['rating_threshold', 'margin'], train_params = trainer_params)

    shutil.rmtree(params['out_dir'])

def test_eval():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)

    trainer_params = {'model': m, 'target_features':[_FEATURES[0]], 'early_stop':False, 'loss_type': 'bce', 'optim_type':'adamw','learning_rate':0.0001, 'tf_learning_rate': 0.00001, 'scheduler_type':None}
    #initialize with valid params 
    t = Trainer(**trainer_params)
    
    df = data_dictionary()
    config = create_config()

    # AUDIO ONLY
    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config, target_labels=[_FEATURES[0]], bucket=None, extension='flac')
    
    #train_loader
    train_loader = DataLoader(d, batch_size=1, shuffle=True, collate_fn=collate_features)
    #val_loader
    val_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_features)
    #test_loader
    test_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_features)
    #fit with val loader - check that train log is in output, run for 7 epochs, assert there is a checkpoint for 0,5 and final for e = 6
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    
    t.test(test_loader = test_loader)
    name_prefix = f'{t.name_prefix}_e{t.epochs}'
    path = m.out_dir /name_prefix
    assert (path/ 'evaluation.json').exists()
    with open(str(path / 'evaluation.json'), 'r') as f:
        data = json.load(f)
    assert 'loss' in data and 'feature_metrics' in data
    feats = data['feature_metrics']
    assert all([f in feats for f in t.target_features])
    feat1 = feats[_FEATURES[0]]
    assert 'bacc' in feat1 and 'acc' in feat1 and 'roc_auc' in feat1
    shutil.rmtree(params['out_dir'])

@pytest.mark.gcs
def test_fit_gcs():
    gcs_prefix, ckpt_prefix, bucket = load_json()
    params = {'model_type': 'wavlm-base', 'out_dir':f'{gcs_prefix}test_model', 'pt_ckpt': ckpt_prefix, 'from_hub': False, 'bucket':bucket, 'delete_download': False, 'finetune_method':'lora'}

    #base test
    m = HFModel(**params)

    trainer_params = {'model': m, 'target_features':[_FEATURES[0]], 'early_stop':False, 'loss_type':'bce', 'scheduler_type':None, 'optim_type':'adamw', 'learning_rate':0.001, 'tf_learning_rate':0.0001}
    #initialize with valid params 
    t = Trainer(**trainer_params)
    
    df = data_dictionary()
    config = create_config()

    # AUDIO ONLY
    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config, target_labels=[_FEATURES[0]], bucket=None, extension='flac')
    
    #train_loader
    train_loader = DataLoader(d, batch_size=1, shuffle=True, collate_fn=collate_features)
    #val_loader
    val_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_features)
    test_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_features)

    args = ['loss_type', 'optim_type', 'scheduler_type', 'learning_rate', 'tf_learning_rate']

    #fit with val loader - check that train log is in output, run for 7 epochs, assert there is a checkpoint for 0,5 and final for e = 6
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    check_checkpoints(m,name_prefix=f'{t.name_prefix}_e7', checkpoints=[0,5],epochs=7, val=True, args=args, train_params = trainer_params)

    t.test(test_loader = test_loader)
    name_prefix = f'{t.name_prefix}_e{t.epochs}'
    path = f'{m.out_dir}{name_prefix}'
    existing = search_gcs(path, path, m.bucket)
    assert any(['evaluation.json' in e for e in existing]), 'Training log not dumped correctly'
    p = f'{path}/evaluation.json'
    blob = m.bucket.blob(p)
    data = json.loads(blob.download_as_string())
    assert 'loss' in data and 'feature_metrics' in data
    feats = data['feature_metrics']
    assert all([f in feats for f in t.target_features])
    feat1 = feats[_FEATURES[0]]
    assert 'bacc' in feat1 and 'acc' in feat1 and 'roc_auc' in feat1
    remove_gcs_directories(gcs_prefix, bucket, 'test_model')  
