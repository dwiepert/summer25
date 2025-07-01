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
from summer25.dataset import WavDataset, collate_waveform
from summer25.training import Trainer
from summer25.constants import _FEATURES

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

    #try different schedulers with specific params

    ##exponential
    trainer_params['scheduler_type'] = 'exponential'
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['end_lr'] = 0.001
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['epochs'] = 1
    t = Trainer(**trainer_params)

    ##warmup multistep
    trainer_params['scheduler_type'] = 'warmup-multistep'
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['warmup_epochs'] = 1 
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['multistep_milestones'] = [0,1]
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    trainer_params['gamma'] = 0.1 
    

    ## multistep
    trainer_params['scheduler_type'] = 'multistep' 
    t = Trainer(**trainer_params)
    del trainer_params['gamma']
    with pytest.raises(AssertionError):
        t = Trainer(**trainer_params)
    del trainer_params['multistep_milestones']
    with pytest.raises(AssertionError):
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

    trainer_params = {'model': m, 'target_features':[_FEATURES[0]], 'early_stop':False}
    #initialize with valid params 
    t = Trainer(**trainer_params)
    
    df = data_dictionary()
    config = create_config()

    # AUDIO ONLY
    feature_extractor = HFExtractor(model_type='wavlm-base')
    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config, target_labels=[_FEATURES[0]], bucket=None, feature_extractor=feature_extractor, extension='flac')
    
    #train_loader
    train_loader = DataLoader(d, batch_size=1, shuffle=True, collate_fn=collate_waveform)
    #val_loader
    val_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_waveform)

    #fit without val loader - check that train log is in output, run for 7 epochs, assert there is a checkpoint for 0,5 and final for e = 6
    t.fit(train_loader=train_loader, epochs=7)
    out_path = m.out_dir / 'weights'
    assert (m.out_dir / 'train_log.json').exists(), 'Training log not dumped correctly'
    assert (out_path / f'checkpoint0_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    assert (out_path / f'checkpoint5_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    assert (out_path / f'final6_{m.model_name}').exists(), 'Final model not saved correctly'
    with open(str(m.out_dir / 'train_log.json'), 'r') as f:
        data = json.load(f)
    assert len(data['train_loss']) == 7
    assert data['val_loss'] == []

    #fit with val loader - check that train log is in output, run for 7 epochs, assert there is a checkpoint for 0,5 and final for e = 6
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    out_path = m.out_dir / 'weights'
    assert (m.out_dir / 'train_log.json').exists(), 'Training log not dumped correctly'
    assert (out_path / f'checkpoint0_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    assert (out_path / f'checkpoint5_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    assert (out_path / f'final6_{m.model_name}').exists(), 'Final model not saved correctly'
    with open(str(m.out_dir / 'train_log.json'), 'r') as f:
        data = json.load(f)
    assert len(data['train_loss']) == 7
    assert len(data['val_loss']) == 7
    shutil.rmtree(params['out_dir'])

def test_earlystop():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)

    trainer_params = {'model': m, 'target_features':[_FEATURES[0]], 'early_stop':True}
    #fit with early stopping
    trainer_params['test'] = True
    t = Trainer(**trainer_params)

    df = data_dictionary()
    config = create_config()
    # AUDIO ONLY
    feature_extractor = HFExtractor(model_type='wavlm-base')
    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config, target_labels=[_FEATURES[0]], bucket=None, feature_extractor=feature_extractor, extension='flac')
    #train_loader
    train_loader = DataLoader(d, batch_size=21, shuffle=True, collate_fn=collate_waveform)
    #val_loader
    val_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_waveform)

    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    out_path = m.out_dir / 'weights'
    assert (m.out_dir / 'train_log.json').exists(), 'Training log not dumped correctly'
    assert (out_path / f'best0_{m.model_name}').exists(), 'Final model not saved correctly'
    with open(str(m.out_dir / 'train_log.json'), 'r') as f:
        data = json.load(f)
    assert len(data['train_loss']) == 1
    assert len(data['val_loss']) == 1
    shutil.rmtree(params['out_dir'])

def test_schedulers():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)

    trainer_params = {'model': m, 'target_features':[_FEATURES[0]], 'early_stop':False}
    #initialize with valid params 
    df = data_dictionary()
    config = create_config()

    # AUDIO ONLY
    feature_extractor = HFExtractor(model_type='wavlm-base')
    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config, target_labels=[_FEATURES[0]], bucket=None, feature_extractor=feature_extractor, extension='flac')
    
    #train_loader
    train_loader = DataLoader(d, batch_size=1, shuffle=True, collate_fn=collate_waveform)
    #val_loader
    val_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_waveform)

    #exponential
    trainer_params['end_lr'] = 0.001
    trainer_params['epochs'] = 7
    trainer_params['scheduler'] = 'exponential'
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    out_path = m.out_dir / 'weights'
    assert (m.out_dir / 'train_log.json').exists(), 'Training log not dumped correctly'
    assert (out_path / f'checkpoint0_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    assert (out_path / f'checkpoint5_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    assert (out_path / f'final6_{m.model_name}').exists(), 'Final model not saved correctly'
    with open(str(m.out_dir / 'train_log.json'), 'r') as f:
        data = json.load(f)
    assert len(data['train_loss']) == 7
    assert len(data['val_loss']) == 7


    #warmup-multistep
    trainer_params['scheduler_type'] = 'warmup-multistep'
    trainer_params['warmup_epochs'] = 2
    trainer_params['multistep_milestones'] = [3,4]
    trainer_params['gamma'] = 0.1 
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    out_path = m.out_dir / 'weights'
    assert (m.out_dir / 'train_log.json').exists(), 'Training log not dumped correctly'
    assert (out_path / f'checkpoint0_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    assert (out_path / f'checkpoint5_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    assert (out_path / f'final6_{m.model_name}').exists(), 'Final model not saved correctly'
    with open(str(m.out_dir / 'train_log.json'), 'r') as f:
        data = json.load(f)
    assert len(data['train_loss']) == 7
    assert len(data['val_loss']) == 7

    #multistep 
    trainer_params['scheduler_type'] = 'multistep'
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    out_path = m.out_dir / 'weights'
    assert (m.out_dir / 'train_log.json').exists(), 'Training log not dumped correctly'
    assert (out_path / f'checkpoint0_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    assert (out_path / f'checkpoint5_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    assert (out_path / f'final6_{m.model_name}').exists(), 'Final model not saved correctly'
    with open(str(m.out_dir / 'train_log.json'), 'r') as f:
        data = json.load(f)
    assert len(data['train_loss']) == 7
    assert len(data['val_loss']) == 7

    #test all assertionsxf
    shutil.rmtree(params['out_dir'])

def test_loss():
    params = {'out_dir':Path('./out_dir'), 'binary':False}

    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)

    trainer_params = {'model': m, 'target_features':[_FEATURES[0]], 'early_stop':False, 'loss_type':'rank'}
    #initialize with valid params
    with pytest.raises(AssertionError): 
        t = Trainer(**trainer_params)
    trainer_params['rating_threshold'] = 2
    t = Trainer(**trainer_params)
    
    df = data_dictionary()
    config = create_config()

    # AUDIO ONLY
    feature_extractor = HFExtractor(model_type='wavlm-base')
    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config, target_labels=[_FEATURES[0]], bucket=None, feature_extractor=feature_extractor, extension='flac')
    
    #train_loader
    train_loader = DataLoader(d, batch_size=1, shuffle=True, collate_fn=collate_waveform)
    #val_loader
    val_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_waveform)

    #fit with val loader - check that train log is in output, run for 7 epochs, assert there is a checkpoint for 0,5 and final for e = 6
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    out_path = m.out_dir / 'weights'
    assert (m.out_dir / 'train_log.json').exists(), 'Training log not dumped correctly'
    assert (out_path / f'checkpoint0_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    assert (out_path / f'checkpoint5_{m.model_name}').exists(), 'Model checkpoint not saved correctly'
    assert (out_path / f'final6_{m.model_name}').exists(), 'Final model not saved correctly'
    with open(str(m.out_dir / 'train_log.json'), 'r') as f:
        data = json.load(f)
    assert len(data['train_loss']) == 7
    assert len(data['val_loss']) == 7
    shutil.rmtree(params['out_dir'])

def test_eval():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)

    trainer_params = {'model': m, 'target_features':[_FEATURES[0]], 'early_stop':False}
    #initialize with valid params 
    t = Trainer(**trainer_params)
    
    df = data_dictionary()
    config = create_config()

    # AUDIO ONLY
    feature_extractor = HFExtractor(model_type='wavlm-base')
    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config, target_labels=[_FEATURES[0]], bucket=None, feature_extractor=feature_extractor, extension='flac')
    
    #train_loader
    train_loader = DataLoader(d, batch_size=1, shuffle=True, collate_fn=collate_waveform)
    #val_loader
    val_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_waveform)
    #test_loader
    test_loader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=collate_waveform)
    #fit with val loader - check that train log is in output, run for 7 epochs, assert there is a checkpoint for 0,5 and final for e = 6
    t = Trainer(**trainer_params)
    t.fit(train_loader=train_loader,val_loader=val_loader, epochs=7)
    
    t.test(test_loader = test_loader)
    assert (m.out_dir / 'evaluation.json').exists()
    with open(str(m.out_dir / 'evaluation.json'), 'r') as f:
        data = json.load(f)
    assert 'loss' in data and 'feature_metrics' in data
    feats = data['feature_metrics']
    assert all([f in feats for f in t.target_features])
    feat1 = feats[_FEATURES[0]]
    assert 'bacc' in feat1 and 'acc' in feat1 and 'roc_auc' in feat1
    shutil.rmtree(params['out_dir'])
