"""
Testing for data splits

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
import json
import os
from pathlib import Path
import shutil

##third-party
import pandas as pd
import pytest

##local
from summer25.constants import _FEATURES
from summer25.dataset import seeded_split
from summer25.io import search_gcs

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
        if i%2 == 0:
            feat1_list.append(3.0)
            feat2_list.append(2.0)
        else:
            feat1_list.append(1.0)
            feat2_list.append(1.0)

    sub_list[10] = sub_list[9]
    date_list[10] = '2024-01-01'
    task_name[20] = 'word_repetition'
    task_name[25] = 'word_repetition'
    task_name[31] = 'word_repetition'
    task_name[40] = 'word_repetition'
    task_name[59] = 'word_repetition'
    task_name[60] = 'word_repetition'
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

def data_dictionary2():
    sub_list = []
    aud_list = []
    date_list = []
    task_name = []
    feat1_list = []
    feat2_list = []

    for i in range(100):
        sub_list.append(f'sub{i}')
        aud_list.append(f'test{i}')
        date_list.append('2025-01-01')
        task_name.append('sentence_repetition')
        if i%2 == 0:
            feat1_list.append(3.0)
            feat2_list.append(2.0)
        else:
            feat1_list.append(1.0)
            feat2_list.append(1.0)

    sub_list[10] = sub_list[9]
    date_list[10] = '2024-01-01'
    task_name[20] = 'word_repetition'
    task_name[25] = 'word_repetition'
    task_name[31] = 'word_repetition'
    task_name[40] = 'word_repetition'
    task_name[59] = 'word_repetition'
    task_name[60] = 'word_repetition'
    data_dict = {'subject': sub_list, 
                'task_name': task_name,
                'incident_date': date_list, 
                'original_audio_id': aud_list, 
                _FEATURES[0]: feat1_list,
                _FEATURES[2]: feat2_list}

    return data_dict

def load_json():
    with open('./private_loading/gcs.json', 'r') as file:
        data = json.load(file)

    gcs_prefix = data['gcs_prefix']
    bucket_name = data['bucket_name']
    project_name = data['project_name']
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.get_bucket(bucket_name)
    return gcs_prefix, bucket

def create_gcs_directories(gcs_prefix, audio=True, split=True, make=True):
    audio_dir = gcs_prefix + 'test_audio' 
    split_dir = gcs_prefix + 'test_split'

    out = [audio_dir, split_dir]
    if not audio_dir:
        out[0] = None 
    
    if not split_dir:
        out[1] = None

    return out

def remove_gcs_directories(gcs_prefix,bucket, directory='test_split', pattern="*"):
    dir = gcs_prefix + f'{directory}'
    existing = search_gcs(pattern, dir, bucket)
    for e in existing:
        blob = bucket.blob(e)
        blob.delete()
    existing = search_gcs(pattern, dir, bucket)
    assert existing == []

def create_gcs_data(directory, name, bucket):
    data_dict = data_dictionary2()
    data_df = pd.DataFrame(data_dict)
    save1 = f'./{name}.csv'
    save2 = f'./{name}.json'
    data_df.to_csv(str(save1), index=False)
    upload_to_gcs(directory, save1, bucket)
    os.remove(save1)
    with open(str(save2), 'w') as f:
        json.dump(data_dict, f, indent=4)
    upload_to_gcs(directory, save2, bucket)
    os.remove(save2)

    existing = search_gcs('*', directory, bucket)
    assert f'{directory}/{name}.csv' in existing, 'Test files not saved.'
    assert f'{directory}/{name}.json' in existing,'Test files not saved.'

def _load(p, bucket, as_json):
    """
    TODO
    """
    if bucket and as_json:
        blob = bucket.blob(p)
        df = pd.DataFrame(json.loads(blob.download_as_string()))
    elif bucket:
        df = pd.read_csv(f'gs://{bucket.name}/{str(p)}')
    elif as_json:
        with open(str(p), 'r') as file:
            df = pd.DataFrame(json.load(file))
    else:
        df = pd.read_csv(str(p))
    return df

def df_compare(search_dir, add, out, as_json=True, bucket=None):
    
    np = Path(search_dir) / add
    if not bucket:
        assert np.exists(), f'{add} does not exist'

    df = _load(str(np), bucket, as_json)

    df['incident_date'] = pd.to_datetime(df['incident_date'])
    out['incident_date'] = pd.to_datetime(out['incident_date'])
    
    try:
        pd.testing.assert_frame_equal(df, out)
    except:
        pd.testing.assert_frame_equal(df.reset_index(drop=True), out.reset_index(drop=True))

### TESTS ###
def test_directories_errors():
    audio_dir1, split_dir1 = create_directories(True, True, False)

    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    #audio_dir/split_dir None
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=None, split_dir=None,**keys)
    
    #audio_dir not None, doesn't exist
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1,**keys)
    
    #audio_dir None, split_dir not None but doesn't exist
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1,**keys)


def test_load_existing_success():
    _ , split_dir1 = create_directories(False, True, True)
    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    #train/test only
    create_data(split_dir1, 'train')
    create_data(split_dir1, 'test')
    out = seeded_split(split_dir=split_dir1, load_existing=True,**keys)
    assert out[1] is None, 'Val is not none'
    out = seeded_split(split_dir=split_dir1, load_existing=True, as_json=True,**keys)
    assert out[1] is None, 'Val is not none'
    
    #train/val/test
    create_data(split_dir1, 'val')
    out = seeded_split(split_dir=split_dir1, load_existing=True,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(split_dir=split_dir1, load_existing=True, as_json=True,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'

    #more than train/test/val shouldn't fail
    create_data(split_dir1, 'other')
    
    out = seeded_split(split_dir=split_dir1, load_existing=True,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(split_dir=split_dir1, load_existing=True, as_json=True,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    
    remove_directories()

def test_load_existing_failure():
    _, split_dir1 = create_directories(False, True, False)
    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    #No split dir and load existing
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1, load_existing=True,**keys)
    
    #split dir exists but no files exist
    split_dir1.mkdir(exist_ok=True)
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1, load_existing=True,**keys)

    #load with invalid files only
    create_data(split_dir1, 'other')
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1, load_existing=True,**keys)
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1, load_existing=True, as_json=True,**keys)

    remove_directories()

@pytest.mark.gcs
def test_load_existing_success_gcs():
    gcs_prefix, bucket = load_json()

    _ , split_dir1 = create_gcs_directories(gcs_prefix, False, True, True)
    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    #train/test only
    create_gcs_data(split_dir1, 'train', bucket)
    create_gcs_data(split_dir1, 'test', bucket)
    out = seeded_split(split_dir=split_dir1, load_existing=True, bucket=bucket, **keys)
    assert out[1] is None, 'Val is not none'
    out = seeded_split(split_dir=split_dir1, load_existing=True, as_json=True, bucket=bucket, **keys)
    assert out[1] is None, 'Val is not none'
    
    #train/val/test
    create_gcs_data(split_dir1, 'val',bucket)
    out = seeded_split(split_dir=split_dir1, load_existing=True, bucket=bucket,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(split_dir=split_dir1, load_existing=True, as_json=True, bucket=bucket,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'

    #more than train/test/val shouldn't fail
    create_gcs_data(split_dir1, 'other',bucket)
    
    out = seeded_split(split_dir=split_dir1, load_existing=True,bucket=bucket,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(split_dir=split_dir1, load_existing=True, as_json=True,bucket=bucket,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    
    remove_gcs_directories(gcs_prefix, bucket=bucket)

def test_load_existing_failure_gcs():
    gcs_prefix, bucket = load_json()
    _ , split_dir1 = create_gcs_directories(gcs_prefix, False, True, True)
    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    #split dir but no files exist
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1, load_existing=True,bucket=bucket, **keys)
    
        
    #load with invalid files only
    create_gcs_data(split_dir1, 'other',bucket)
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1, load_existing=True,bucket=bucket,**keys)
    
        
    with pytest.raises(AssertionError):
        seeded_split(split_dir=split_dir1, load_existing=True, as_json=True,bucket=bucket,**keys)
    
        
    remove_gcs_directories(gcs_prefix, bucket=bucket)

def test_metadata_load():
    audio_dir1, split_dir1 = create_directories(True, True, True)
    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    #test 1 - no metadata file in audio dir
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False,**keys)
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True,**keys)

    create_data(audio_dir1, 'metadata')
    
    #test 2 - load metadata from csv
    out = seeded_split(audio_dir=audio_dir1, split_dir=None, as_json=False,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'

    #test 3 - load metadata from json
    out = seeded_split(audio_dir=audio_dir1, split_dir=None, as_json=True,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'

    #test 4 - more than one metadatafile in audio dir
    create_data(audio_dir1, 'metadata1')
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False,**keys)
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True,**keys)
    
    os.remove(str(audio_dir1 / 'metadata1.csv'))
    os.remove(str(audio_dir1 / 'metadata1.json'))

    #test 5 - no date key
    keys['date_key'] = 'date'
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False,**keys)
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True,**keys)

    #test 6 - no subject key
    keys['date_key'] = 'incident_date'
    keys['subject_key'] = 'date'
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False,**keys)
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True,**keys)

    #test 7 - no audio key
    keys['subject_key'] = 'subject'
    keys['audio_key'] = 'date'
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False,**keys)
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True,**keys)
    
    #test 9 - target tasks not none, task key not in col
    keys['audio_key'] = 'original_audio_id'
    keys['task_key'] = 'date'
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_tasks=['sentence_repetition'],**keys)
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, target_tasks=['sentence_repetition'],**keys)

    remove_directories()

@pytest.mark.gcs
def test_create_gcs():
    gcs_prefix, bucket = load_json()
    audio_dir1, split_dir1 = create_gcs_directories(gcs_prefix, False, True, True)
    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    #test 1 - no metadata file in audio dir
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False,bucket=bucket,**keys)
       
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True,bucket=bucket,**keys)   
    
    create_gcs_data(audio_dir1, 'metadata',bucket)
    
    #test 2 - load metadata from csv
    out = seeded_split(audio_dir=audio_dir1, split_dir=None, as_json=False, bucket=bucket,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, bucket=bucket,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'

    #test 3 - load metadata from json
    out = seeded_split(audio_dir=audio_dir1, split_dir=None, as_json=True, bucket=bucket,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, bucket=bucket,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'

    #test 4 - more than one metadatafile in audio dir
    create_gcs_data(audio_dir1, 'metadata1',bucket)
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, bucket=bucket,**keys)
    
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, bucket=bucket,**keys)
    
    remove_gcs_directories(gcs_prefix, bucket, Path(audio_dir1).name, pattern='metadata')


def test_metadata_filtering():
    audio_dir1, split_dir1 = create_directories(True, True, True)
    create_data(audio_dir1, 'metadata')
    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    #no target tasks
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False,**keys)
    vals = out[0]['task_name'].values.tolist()
    assert any([v == 'sentence_repetition' for v in vals]) and any([v == 'word_repetition' for v in vals]), f'Did not keep all tasks'

    #invalid task
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_tasks=['sentence'],**keys)
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, target_tasks=['sentence'],**keys)

    #valid task
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_tasks=['sentence_repetition'],**keys)
    vals = out[0]['task_name'].values.tolist()
    assert all([v == 'sentence_repetition' for v in vals]), 'Missing target task'
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_tasks=['sentence_repetition', 'word_repetition'],**keys)
    vals = out[0]['task_name'].values.tolist()
    assert any([v == 'sentence_repetition' for v in vals]) and any([v == 'word_repetition' for v in vals]), 'Missing target task'

    #no target features
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False,**keys)
    assert all([_FEATURES[0] in o.columns.to_list() for o in out]) and all([_FEATURES[2] in o.columns.to_list() for o in out])
    
    #invalid target feature
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_features = ['test'],**keys)
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, target_features = ['test'],**keys)

    #valid target features
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_features=[_FEATURES[0]],**keys)
    assert all([_FEATURES[0] in o.columns.to_list() for o in out]), 'Missing target features'
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_features=[_FEATURES[0],_FEATURES[2]],**keys)
    assert all([_FEATURES[0] in o.columns.to_list() for o in out]) and all([_FEATURES[2] in o.columns.to_list() for o in out]), 'Missing target features'

    #target feature missing from column names
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, target_features = [_FEATURES[1]],**keys)
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, target_features = [_FEATURES[1]],**keys)

    # random 
    # give audio dir only + load existing
    out = seeded_split(audio_dir=audio_dir1, load_existing=True,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(audio_dir=audio_dir1, load_existing=False,**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'
    out = seeded_split(audio_dir=str(audio_dir1),**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'

    remove_directories()

def test_proportions():
    audio_dir1, split_dir1 = create_directories(True, True, True)
    create_data(audio_dir1, 'metadata')
    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    #more than 3 proportions, less than 3 proportions
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[.7,.1,.1,.1],**keys)
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, proportions=[.7,.1,.1,.1],**keys)

    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[0,.7,.3],**keys)
    with pytest.raises(ValueError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, proportions=[0,.7,.3],**keys)
    
    #train, test, or val only
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[1,0,0],**keys)
    assert out[0] is not None and out[1] is None and out[2] is None, 'Only one output should have values'
    size0 = len(out[0])
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[0,1,0],**keys)
    assert out[0] is None and out[1] is not None and out[2] is None,'Only one output should have values'
    size1 = len(out[1])
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[0,0,1],**keys)
    assert out[0] is None and out[1] is None and out[2] is not None,'Only one output should have values'
    size2 = len(out[2])
    assert size0 == size1 and size1 == size2, 'All outputs should have the same size'
    
    #train/test only or train/val only
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[.7,.3,0],**keys)
    assert out[0] is not None and out[1] is not None and out[2] is None, 'Only one ouput should be None'
    size_big0 = len(out[0])
    size_small0 = len(out[1])

    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[.7,0,.3],**keys)
    assert out[0] is not None and out[1] is None and out[2] is not None, 'Only one ouput should be None'
    size_big1 = len(out[0])
    size_small1 = len(out[2])
    assert size_big0 == size_big1 and size_small0 == size_small1, 'Output sizes should match'
    
    #train/test/val 
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[.7,.15,.15],**keys)
    assert all([o is not None for o in out]), 'There should be no None outputs'

    #proportions add up to 1 
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[.7,.1,.1],**keys)
    with pytest.raises(AssertionError):
        seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, proportions=[.7,.1,.1],**keys)
    
    remove_directories()

def test_output():
    #TODO: make sure rank cols are included and have values beyond 0,1,nan + ensure values aren't lost
    audio_dir1, split_dir1 = create_directories(True, True, True)
    create_data(audio_dir1, 'metadata')
    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    #test 21 - all expected columns in final dataframes (NEED ORIGINAL AUDIO ID TO BE ADDED!!!! something off with pooled annotations)
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, proportions=[.7,.15,.15],**keys)
    size = len(data_dictionary()['subject'])
    tr_size = int(size*.7)
    v_size = int(size*.15)
    te_size = int(size*.15)
    if (tr_size+v_size+te_size) != size:
        te_size += 1
    assert (tr_size+v_size+te_size) == size

    assert all([o is not None for o in out]), 'There should be no None outputs'
    rank_features = [f'rank_{feat}' for feat in [_FEATURES[0],_FEATURES[2]]]
    expected_columns = ['subject', 'original_audio_id', 'task_name', 'incident_date', _FEATURES[0], _FEATURES[2]] + rank_features
    tasks = []
    for o in out:
        for e in expected_columns:
            assert e in o.columns.to_list(), f'{e} missing in output.'
        vals = o['task_name'].values.tolist()
        tasks.extend(vals)
        #check features are binarized
        vals2 = o[_FEATURES[0]].values.tolist()
        feat = _FEATURES[0]
        vals3 = o[f'rank_{feat}'].values.tolist()
        assert all([(v == 0 or v == 1) for v in vals2]), 'Unrecognized value in feature column'
        #check 3s maintained for rank cols
        assert all([(v==0.0 or v == 3.0) for v in vals3]), 'Unrecognized value in feature column'
    assert any([v == 'sentence_repetition' for v in tasks]) and any([v == 'word_repetition' for v in tasks]), f'Did not keep all tasks'
    assert len(out[0]) == tr_size
    assert len(out[1]) == v_size
    assert len(out[2]) == te_size
    
    remove_directories()

def test_saving():
    audio_dir1, split_dir1 = create_directories(True, True, True)
    create_data(audio_dir1, 'metadata')
    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    #test 22 - test saving
    name = f'{audio_dir1.name}_seed{42}_tr{.7}v{.15}te{.15}_ntasks2_nfeats2'
    search_dir = split_dir1 / name
    
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, save=True, seed=42, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"],**keys)
    assert search_dir.exists(), 'Save dir does not exist'
    
    df_compare(search_dir, 'train.json',out[0])
    df_compare(search_dir, 'val.json', out[1])
    df_compare(search_dir, 'test.json',out[2])

    shutil.rmtree(split_dir1)
    split_dir1.mkdir(exist_ok=True)
    
    #csv
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, save=True, seed=42, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"],**keys)
    assert search_dir.exists(), 'Save dir does not exist'
    df_compare(search_dir, 'train.csv',out[0], False)
    df_compare(search_dir, 'val.csv', out[1], False)
    df_compare(search_dir, 'test.csv',out[2], False)

    remove_directories()

@pytest.mark.gcs
def test_saving_gcs():
    gcs_prefix, bucket = load_json()
    audio_dir1, split_dir1 = create_gcs_directories(gcs_prefix, False, True, True)
    create_gcs_data(audio_dir1, 'metadata',bucket)

    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    #test 22 - test saving
    name = f'{Path(audio_dir1).name}_seed{42}_tr{.7}v{.15}te{.15}_ntasks2_nfeats2'
    search_dir = f'{split_dir1}/{name}'
    
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=True, save=True, seed=42, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"], bucket=bucket, **keys)
    existing = search_gcs('*', split_dir1, bucket)
    assert existing != [], 'Split not properly saved.'
    
    df_compare(search_dir, 'train.json',out[0], as_json=True, bucket=bucket)
    df_compare(search_dir, 'val.json', out[1], as_json=True, bucket=bucket)
    df_compare(search_dir, 'test.json',out[2], as_json=True, bucket=bucket)

    remove_gcs_directories(gcs_prefix, bucket)
    
    #csv
    out = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, save=True, seed=42, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"], bucket=bucket,**keys)
    existing = search_gcs('*', split_dir1, bucket)
    assert existing != [], 'Split not properly saved.'
    
    df_compare(search_dir, 'train.csv',out[0], as_json=False, bucket=bucket)
    df_compare(search_dir, 'val.csv', out[1], as_json=False, bucket=bucket)
    df_compare(search_dir, 'test.csv',out[2], as_json=False, bucket=bucket)

    remove_gcs_directories(gcs_prefix, bucket)

def test_seeding():
    audio_dir1, split_dir1 = create_directories(True, True, True)
    create_data(audio_dir1, 'metadata')
    keys = {'subject_key': 'subject', 'date_key':'incident_date', 'audio_key':'original_audio_id', 'task_key':'task_name'}
    out42_1 = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, save=True, seed=42, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"],**keys)
    out42_2 = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, save=True, seed=42, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"],**keys)
    
    out100_1 = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, save=True, seed=100, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"],**keys)
    out100_2 = seeded_split(audio_dir=audio_dir1, split_dir=split_dir1, as_json=False, save=True, seed=100, proportions=[.7,.15,.15], target_features=[_FEATURES[0],_FEATURES[2]], target_tasks=["sentence_repetition", "word_repetition"],**keys)

    for i in range(len(out42_1)):
        assert out42_1[i].equals(out42_2[i]), 'Same seed not producing same results'
        assert out100_1[i].equals(out100_2[i]), 'Same seed not producing same results'
        assert (out100_1[i].equals(out42_1[i])) is False, 'Different seeds producing same results.'
    
    remove_directories()

