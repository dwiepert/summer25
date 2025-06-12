from pathlib import Path 
import pytest
import shutil
import os
import pandas as pd
from summer25.dataset import seeded_split
from summer25.constants import _FEATURES
import json


#### SUPPORT CODE ### 
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

    return data_dict

def create_directories(audio=True, split=True, make=True):
    audio_dir = Path('./example_audio')
    split_dir = Path('./example_split')

    if split_dir.exists():
        shutil.rmtree(split_dir)
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    
    out = [audio_dir, split_dir]
    if not audio_dir:
        out[0] = None 
    elif make:
        audio_dir.mkdir(exist_ok=True)
    
    if not split_dir:
        out[1] = None
    elif make:
        split_dir.mkdir(exist_ok=True)
    return out

def remove_directories():
    audio_dir = Path('./example_audio')
    split_dir = Path('./example_split')

    if split_dir.exists():
        shutil.rmtree(split_dir)
    if audio_dir.exists():
        shutil.rmtree(audio_dir)

def create_data(directory, name):
    data_dict = data_dictionary()
    data_df = pd.DataFrame(data_dict)
    new1 = directory / (name+'.csv')
    new2 = directory / (name + '.json')
    data_csv = data_df.to_csv(str(new1))
    with open(str(new2), 'w') as f:
        json.dump(data_dict, f, indent=4)

    assert new1.exists() and new2.exists(), 'Test files not saved.'

def df_compare(search_dir, add, out, as_json=True):
    np = search_dir / add
    assert np.exists(), f'{add} does not exist'

    if as_json:
        with open(str(np), 'r') as f:
            df = pd.DataFrame(json.load(f))
    else:
        df = pd.read_csv(np)

    df['incident_date'] = pd.to_datetime(df['incident_date'])
    out['incident_date'] = pd.to_datetime(out['incident_date'])
    
    try:
        pd.testing.assert_frame_equal(df, out)
    except:
        pd.testing.assert_frame_equal(df.reset_index(drop=True), out.reset_index(drop=True))

### TESTS ###
def test_directories_errors():
    audio_dir1, split_dir1 = create_directories(True, True, False)

    #audio_dir/split_dir None
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=None, split_dir=None)
    
    #audio_dir not None, doesn't exist
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1)
    
    #audio_dir None, split_dir not None but doesn't exist
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1)


def test_load_existing_success():
    _ , split_dir1 = create_directories(False, True, True)

    #train/test only
    create_data(split_dir1, 'train')
    create_data(split_dir1, 'test')
    out = seeded_split(split_dir=split_dir1, load_existing=True)
    assert out[1] is None, 'Val is not none'
    out = seeded_split(split_dir=split_dir1, load_existing=True, as_json=True)
    assert out[1] is None, 'Val is not none'
    
    #train/val/test
    create_data(split_dir1, 'val')
    out = seeded_split(split_dir=split_dir1, load_existing=True)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(split_dir=split_dir1, load_existing=True, as_json=True)
    assert all([o is not None for o in out]), 'There should be no None outputs'

    #more than train/test/val shouldn't fail
    create_data(split_dir1, 'other')
    
    out = seeded_split(split_dir=split_dir1, load_existing=True)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(split_dir=split_dir1, load_existing=True, as_json=True)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    
    remove_directories()

def test_load_existing_failure():
    _, split_dir1 = create_directories(False, True, False)
    
    #No split dir and load existing
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1, load_existing=True)
    
    #split dir exists but no files exist
    split_dir1.mkdir(exist_ok=True)
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1, load_existing=True)

    #load with invalid files only
    create_data(split_dir1, 'other')
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1, load_existing=True)
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1, load_existing=True, as_json=True)

    remove_directories()

def test_metadata_load():
    audio_dir1, split_dir1 = create_directories(True, True, True)

    #test 1 - no metadata file in audio dir
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False)
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True)

    create_data(audio_dir1, 'metadata')
    
    #test 2 - load metadata from csv
    out = seeded_split(audio_dir=audio_dir1, split_dir=None, as_json=False)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False)
    assert all([o is not None for o in out]), 'There should be no None outputs'

    #test 3 - load metadata from json
    out = seeded_split(audio_dir=audio_dir1, split_dir=None, as_json=True)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True)
    assert all([o is not None for o in out]), 'There should be no None outputs'

    #test 4 - more than one metadatafile in audio dir
    create_data(audio_dir1, 'metadata1')
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False)
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True)
    
    os.remove(str(audio_dir1 / 'metadata1.csv'))
    os.remove(str(audio_dir1 / 'metadata1.json'))

    #test 5 - no date key
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, date_key="date")
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, date_key="date")

    #test 6 - no subject key
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, subject_key="date")
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, subject_key="date")

    #test 7 - no audio key
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, audio_key="date")
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, audio_key="date")
    
    #test 9 - target tasks not none, task key not in col
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, task_key="date", target_tasks=['sentence_repetition'])
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, task_key="date", target_tasks=['sentence_repetition'])

    remove_directories()

def test_metadata_filtering():
    audio_dir1, split_dir1 = create_directories(True, True, True)
    create_data(audio_dir1, 'metadata')

    #no target tasks
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False)
    vals = out[0]['task_name'].values.tolist()
    assert any([v == 'sentence_repetition' for v in vals]) and any([v == 'word_repetition' for v in vals]), f'Did not keep all tasks'

    #invalid task
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_tasks=['sentence'])
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, target_tasks=['sentence'])

    #valid task
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_tasks=['sentence_repetition'])
    vals = out[0]['task_name'].values.tolist()
    assert all([v == 'sentence_repetition' for v in vals]), 'Missing target task'
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_tasks=['sentence_repetition', 'word_repetition'])
    vals = out[0]['task_name'].values.tolist()
    assert any([v == 'sentence_repetition' for v in vals]) and any([v == 'word_repetition' for v in vals]), 'Missing target task'

    #no target features
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False)
    assert all([_FEATURES[0] in o.columns.to_list() for o in out]) and all([_FEATURES[2] in o.columns.to_list() for o in out])
    
    #invalid target feature
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_features = ['test'])
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, target_features = ['test'])

    #valid target features
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_features=[_FEATURES[0]])
    assert all([_FEATURES[0] in o.columns.to_list() for o in out]), 'Missing target features'
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_features=[_FEATURES[0],_FEATURES[2]])
    assert all([_FEATURES[0] in o.columns.to_list() for o in out]) and all([_FEATURES[2] in o.columns.to_list() for o in out]), 'Missing target features'

    #target feature missing from column names
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_features = [_FEATURES[1]])
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, target_features = [_FEATURES[1]])

    # random 
    # give audio dir only + load existing
    out = seeded_split(audio_dir=audio_dir1, load_existing=True)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(audio_dir=audio_dir1, load_existing=False)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(audio_dir=str(audio_dir1))
    assert all([o is not None for o in out]), 'There should be no None outputs'

    remove_directories()

def test_proportions():
    audio_dir1, split_dir1 = create_directories(True, True, True)
    create_data(audio_dir1, 'metadata')

    #more than 3 proportions, less than 3 proportions
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[.7,.1,.1,.1])
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, proportions=[.7,.1,.1,.1])

    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[0,.7,.3])
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, proportions=[0,.7,.3])
    
    #train, test, or val only
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[1,0,0])
    assert out[0] is not None and out[1] is None and out[2] is None, 'Only one output should have values'
    size0 = len(out[0])
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[0,1,0])
    assert out[0] is None and out[1] is not None and out[2] is None,'Only one output should have values'
    size1 = len(out[1])
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[0,0,1])
    assert out[0] is None and out[1] is None and out[2] is not None,'Only one output should have values'
    size2 = len(out[2])
    assert size0 == size1 and size1 == size2, 'All outputs should have the same size'
    
    #train/test only or train/val only
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[.7,.3,0])
    assert out[0] is not None and out[1] is not None and out[2] is None, 'Only one ouput should be None'
    size_big0 = len(out[0])
    size_small0 = len(out[1])

    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[.7,0,.3])
    assert out[0] is not None and out[1] is None and out[2] is not None, 'Only one ouput should be None'
    size_big1 = len(out[0])
    size_small1 = len(out[2])
    assert size_big0 == size_big1 and size_small0 == size_small1, 'Output sizes should match'
    
    #train/test/val 
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[.7,.15,.15])
    assert all([o is not None for o in out]), 'There should be no None outputs'

    #proportions add up to 1 
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[.7,.1,.1])
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, proportions=[.7,.1,.1])
    
    remove_directories()

def test_output():
    audio_dir1, split_dir1 = create_directories(True, True, True)
    create_data(audio_dir1, 'metadata')

    #test 21 - all expected columns in final dataframes (NEED ORIGINAL AUDIO ID TO BE ADDED!!!! something off with pooled annotations)
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[.7,.15,.15])
    size = len(data_dictionary()['subject'])
    tr_size = int(size*.7)
    v_size = int(size*.15)
    te_size = int(size*.15)
    if (tr_size+v_size+te_size) != size:
        te_size += 1
    assert (tr_size+v_size+te_size) == size

    assert all([o is not None for o in out]), 'There should be no None outputs'
    expected_columns = ['subject', 'original_audio_id', 'task_name', 'incident_date', _FEATURES[0], _FEATURES[2]]
    tasks = []
    for o in out:
        for e in expected_columns:
            assert e in o.columns.to_list(), f'{e} missing in output.'
        vals = o['task_name'].values.tolist()
        tasks.extend(vals)
        #check features are binarized
        vals2 = o[_FEATURES[0]].values.tolist()
        assert all([(v == 0 or v == 1) for v in vals2]), 'Unrecognized value in feature column'
    assert any([v == 'sentence_repetition' for v in tasks]) and any([v == 'word_repetition' for v in tasks]), f'Did not keep all tasks'
    assert len(out[0]) == tr_size
    assert len(out[1]) == v_size
    assert len(out[2]) == te_size
    
    remove_directories()

def test_saving():
    audio_dir1, split_dir1 = create_directories(True, True, True)
    create_data(audio_dir1, 'metadata')

    #test 22 - test saving
    name = f'{audio_dir1.name}_seed{42}_tr{.7}v{.15}te{.15}_ntasks2_nfeats2'
    search_dir = split_dir1 / name
    
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, save=True, seed=42, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"])
    assert search_dir.exists(), 'Save dir does not exist'
    
    df_compare(search_dir, 'train.json',out[0])
    df_compare(search_dir, 'val.json', out[1])
    df_compare(search_dir, 'test.json',out[2])

    shutil.rmtree(split_dir1)
    split_dir1.mkdir(exist_ok=True)
    
    #csv
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, save=True, seed=42, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"])
    assert search_dir.exists(), 'Save dir does not exist'
    df_compare(search_dir, 'train.csv',out[0], False)
    df_compare(search_dir, 'val.csv', out[1], False)
    df_compare(search_dir, 'test.csv',out[2], False)

    remove_directories()

def test_seeding():
    audio_dir1, split_dir1 = create_directories(True, True, True)
    create_data(audio_dir1, 'metadata')

    out42_1 = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, save=True, seed=42, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"])
    out42_2 = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, save=True, seed=42, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"])
    
    out100_1 = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, save=True, seed=100, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"])
    out100_2 = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, save=True, seed=100, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"])

    for i in range(len(out42_1)):
        assert out42_1[i].equals(out42_2[i]), 'Same seed not producing same results'
        assert out100_1[i].equals(out100_2[i]), 'Same seed not producing same results'
        assert (out100_1[i].equals(out42_1[i])) is False, 'Different seeds producing same results.'
    
    remove_directories()