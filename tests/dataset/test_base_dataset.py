"""
Testing for Base Dataset

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in

##third-party
import numpy as np
import pandas as pd
import pytest
import torch
import torchvision

##local
from summer25.dataset import BaseDataset
from summer25.constants import _FEATURES

class Practice():
    def __call__(self, sample):
        sample_out = sample.copy()
        sample_out['test'] = [1 for i in range(len(sample['uid']))]
        return sample_out

def data_dictionary():
    sub_list = []
    aud_list = []
    date_list = []
    task_name = []
    feat1_list = []
    feat2_list = []

    for i in range(100):
        sub_list.append(f'sub{i}')
        aud_list.append(f'wav{i}')
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

def test_base_initialization():
    # create metadata in pd dataframe
    df = data_dictionary()

    with pytest.raises(AssertionError):
        d = BaseDataset(data = df, target_labels=None, transforms=None)
    
    with pytest.raises(AssertionError):
        d = BaseDataset(data = np.array([0.0,0.1,1.0]), target_labels=[0,1], transforms=None)
    
    #incorrect target labels - pandas
    target_labels = [0, 1]
    with pytest.raises(AssertionError):
        d = BaseDataset(data = df, target_labels=target_labels, transforms=None)
    target_labels = [_FEATURES[0],_FEATURES[3]]
    with pytest.raises(AssertionError):
        d = BaseDataset(data = df, target_labels=target_labels, transforms=None)

    target_labels = [_FEATURES[0],_FEATURES[2]]
    best = BaseDataset(data = df, target_labels=target_labels, transforms=None)
    out_df = best.get_data()

    try:
        pd.testing.assert_frame_equal(df, out_df)
    except:
        raise ValueError(f'Data frames not equivalent. {out_df.columns}')

    #invalid data types
    np_data = df[[_FEATURES[0],_FEATURES[2]]].to_numpy()
    with pytest.raises(AssertionError):
        d = BaseDataset(data = np_data, target_labels=target_labels, transforms=None)

    list_data = np_data.tolist()
    with pytest.raises(AssertionError):
        d = BaseDataset(data = list_data, target_labels=target_labels, transforms=None)

def test_get_items():
    #no transforms, try with idx that's too high
    df = data_dictionary()
    target_labels = [_FEATURES[0],_FEATURES[2]]
    d = BaseDataset(data=df, target_labels=target_labels, transforms=None)

    #no transforms, try with idx that's too high
    with pytest.raises(IndexError):
        out = d[100]

    #try with single value
    out = d[0]
    test_uid = out['uid']
    assert out['uid'] == 'sub0', f'{test_uid}'
    vals1 = out['targets']
    vals2 = df[target_labels].iloc[0].values
    assert all([vals1[i] == vals2[i] for i in range(len(vals1))]), 'Incorrect values grabbed.'

    #try with list of ints
    with pytest.raises(AssertionError):
        out = d[[0,1]]

    #try with torch tensor
    out = d[torch.tensor([0])]
    given = out['uid']
    expected = 'sub0'
    assert given == expected, 'Incorrect items grabbed.'

    vals1 = out['targets']
    vals2 = df[target_labels].iloc[torch.tensor([0])].values.tolist()
    assert all([vals1[i] == vals2[i] for i in range(len(vals1))]), 'Incorrect values grabbed.'

    #try with invalid index type
    with pytest.raises(IndexError):
        out = d[np.ndarray([0])]
    with pytest.raises(IndexError):
        out = d[['1']]
    with pytest.raises(IndexError):
        out = d[['sub0']]

def test_transforms():
    transforms = torchvision.transforms.Compose([Practice()])
    df = data_dictionary()
    target_labels = [_FEATURES[0],_FEATURES[2]]
    d = BaseDataset(data=df, target_labels=target_labels, transforms=transforms)

    out = d[0]
    assert 'test' in out and all([v == 1 for v in out['test']]), 'Transform applied.'

