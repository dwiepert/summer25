"""
Testing for Base Dataset

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
import json
from pathlib import Path

##third-party
from google.cloud import storage
import pandas as pd
import pytest
import torch
import torchvision

##local
from summer25.dataset import WavDataset
from summer25.transforms import UidToPath
from summer25.constants import _FEATURES
from summer25.models import HFExtractor
from summer25.io import search_gcs

def data_dictionary(aud_name='wav'):
    sub_list = ['1919-142785-0008']
    aud_list = ['1919-142785-0008']
    date_list = ['2025-01-01']
    task_name = ['random']
    feat1_list = [1.0]
    feat2_list = [1.0]

    for i in range(100):
        sub_list.append(f'sub{i}')
        aud_list.append(f'{aud_name}{i}')
        date_list.append('2025-01-01')
        task_name.append('sentence_repetition')
        feat1_list.append(2.0)
        feat2_list.append(2.0)

    sub_list[10] = sub_list[9]
    date_list[10] = '2024-01-01'
    task_name[20] = 'word_repetition'
    data_dict = {'subject': sub_list, 
                'task_name': task_name,
                'incident_date': date_list, 
                'original_audio_id': aud_list, 
                _FEATURES[0]: feat1_list,
                _FEATURES[2]: feat2_list}

    data_df = pd.DataFrame(data_dict)
    data_df = data_df.set_index('subject')
    return data_df

def data_dictionary2(aud_name='wav'):
    sub_list = ['1919-142785-0008']
    aud_list = ['test1']
    date_list = ['2025-01-01']
    task_name = ['random']
    feat1_list = [1.0]
    feat2_list = [1.0]

    for i in range(100):
        sub_list.append(f'sub{i}')
        aud_list.append(f'{aud_name}{i}')
        date_list.append('2025-01-01')
        task_name.append('sentence_repetition')
        feat1_list.append(2.0)
        feat2_list.append(2.0)

    sub_list[10] = sub_list[9]
    date_list[10] = '2024-01-01'
    task_name[20] = 'word_repetition'
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

def test_dataset_initialization():
    #create metadata in pd dataframe
    df = data_dictionary()
    config = create_config()

    waveform_loader = UidToPath(prefix = Path('./tests/audio_examples/'), ext='flac')
    transform = torchvision.transforms.Compose([waveform_loader])

    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config, target_labels=[_FEATURES[0]], bucket=None, transforms=transform)
    assert d.use_librosa is False, 'Did not extract values from config properly'

def test_get_item():
    df = data_dictionary()
    config = create_config()

    # AUDIO ONLY
    d = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='whisper-tiny', config=config, target_labels=[_FEATURES[0]], bucket=None, extension='flac')
    out = d[0]
    assert all([u in out for u in ['uid','targets','waveform','sample_rate']])
    assert isinstance(out['uid'],str)
    assert isinstance(out['targets'],torch.Tensor)
    assert isinstance(out['waveform'],torch.Tensor)
    assert isinstance(out['sample_rate'],int)

    # LIBROSA
    config2 = create_config(use_librosa=True)
    d2 = WavDataset(data=df, prefix='./tests/audio_examples/',uid_col='original_audio_id', model_type='whisper-tiny', config=config2, target_labels=[_FEATURES[0]], bucket=None, extension='flac')
    out = d2[0]
    assert all([u in out for u in ['uid','targets','waveform','sample_rate']])
    assert isinstance(out['uid'],str)
    assert isinstance(out['targets'],torch.Tensor)
    assert isinstance(out['waveform'],torch.Tensor)
    assert isinstance(out['sample_rate'],int)

    #AUGMENT
    config3 = create_config(augment=True)
    d3 = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='whisper-tiny', config=config3, target_labels=[_FEATURES[0]], bucket=None, extension='flac')
    out = d3[0]
    assert all([u in out for u in ['uid','targets','waveform','sample_rate']])
    assert isinstance(out['uid'],str)
    assert isinstance(out['targets'],torch.Tensor)
    assert isinstance(out['waveform'],torch.Tensor)
    assert isinstance(out['sample_rate'],int)

    #give truncate but revise clip length
    config4 = create_config(truncate=True)
    d4 = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='whisper-tiny', config=config4, target_labels=[_FEATURES[0]], bucket=None, extension='flac')
    out = d4[0]
    assert all([u in out for u in ['uid','targets','waveform','sample_rate']])
    assert isinstance(out['uid'],str)
    assert isinstance(out['targets'],torch.Tensor)
    assert isinstance(out['waveform'],torch.Tensor)
    assert isinstance(out['sample_rate'],int)

    #truncate and wavlm-base model
    d5 = WavDataset(data=df, prefix='./tests/audio_examples/', uid_col='original_audio_id', model_type='wavlm-base', config=config4, target_labels=[_FEATURES[0]], bucket=None, extension='flac')
    out = d5[0]
    assert all([u in out for u in ['uid','targets','waveform','sample_rate']])
    assert isinstance(out['uid'],str)
    assert isinstance(out['targets'],torch.Tensor)
    assert isinstance(out['waveform'],torch.Tensor)
    assert isinstance(out['sample_rate'],int)

def load_json():
    with open('./private_loading/gcs.json', 'r') as file:
        data = json.load(file)

    gcs_prefix = data['gcs_prefix']
    bucket_name = data['bucket_name']
    project_name = data['project_name']
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.get_bucket(bucket_name)
    return gcs_prefix, bucket

@pytest.mark.gcs
def test_get_item_gcs():
    df = data_dictionary2(aud_name='test')
    config = create_config()
    gcs_prefix, bucket = load_json()

    d = WavDataset(data=df, prefix=gcs_prefix+'/test_audio', model_type='wavlm-base',  uid_col='original_audio_id', config=config, target_labels=[_FEATURES[0]], bucket=bucket)
    out = d[0]
    assert all([u in out for u in ['uid','targets','waveform','sample_rate']])
    assert isinstance(out['uid'],str)
    assert isinstance(out['targets'],torch.Tensor)
    assert isinstance(out['waveform'],torch.Tensor)
    assert isinstance(out['sample_rate'],int)
