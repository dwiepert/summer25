"""
Split dataset into random train/val/test sets 

Author(s): Hugo Botha, Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
#built-in
import json
import os
from pathlib import Path
from typing import Union,List,Tuple

#third-party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#local
from summer25.constants import _TASKS, _FEATURES
from summer25.io import search_gcs, upload_to_gcs

def _load(p:Union[str, Path], bucket=None, as_json:bool=False) -> pd.DataFrame:
    """
    Load dataframe from any source

    :param p: pathlike - path to dataframe
    :param bucket: google cloud storage bucket (default = None)
    :param as_json: bool, true if loading from json file
    :return df: loaded dataframe
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


def _load_existing(split_dir:Path, as_json:bool, date_key:str, bucket) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load an existing split
    :param split_dir: pathlike, path to directory to save splits to. May have existing splits. 
    :param as_json: bool, specify whether loading/saving should use .json
    :param date_key: str, column name storing dates
    :param bucket: gcs bucket
    :return: train, val, test split dataframes
    """
    poss = ['train', 'val', 'test']
    if as_json:
        pattern = '.json'
    else:
        pattern = '.csv'
    if bucket:
        paths = search_gcs(pattern, str(split_dir), bucket)
    else:
        paths = [j for j in split_dir.rglob(f'*{pattern}')]
        if paths == []:
            return None, None, None
        
    paths = [j for j in paths if Path(j).with_suffix("").name in poss]

    store = {}
    for p in paths:
        df = _load(p, bucket, as_json)
        df[date_key] = pd.to_datetime(df[date_key]).dt.strftime('%Y-%m-%d')
        store[Path(p).with_suffix("").name] = df

    for k in poss:
        if k not in store:
            print(f'No {k} set in split directory. Confirm this is expected behavior.')
            store[k] = None
    
    ks = list(store.keys())

    return store['train'], store['val'], store['test']

def _sklearn_split(split_table:pd.DataFrame, subject_key:str, size:float, seed:int) -> Tuple[pd.Series, pd.Series]:
    """
    Run train/test split

    :param split_table: pd.DataFrame, table to create splits from
    :param subject_key: str, column to split on
    :param size: float, test set size as proportion
    :param seed: int, random seed
    :return test_subjects: pd.Series, column with train subjects
    :return test_subjects: pd.Series, column with test subjects
    """
    # RUN SPLIT
    X_train, X_test, y_train, y_test = train_test_split(split_table[[subject_key]], split_table[[subject_key]], 
                                                        stratify=split_table[['stratify']],test_size=size, random_state=seed)

    train_subjects =  X_train[subject_key].values
    test_subjects = X_test[subject_key].values
    return train_subjects, test_subjects

def seeded_split(subject_key:str, date_key:str, audio_key:str, task_key:str, audio_dir:Union[Path, str]=None, split_dir:Union[Path,str]=None, proportions:List[float]=[.7, .15, .15], seed:int=42,
          save:bool=False, load_existing:bool=False, as_json:bool=False,
          target_tasks:List[str]=None, target_features:List[str] = None, stratify_threshold:int=10,
          bucket=None ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/test/val splits

    :param subject_key: str, name of column/key containing subject identifiers
    :param date_key: str, name of column/key containing date
    :param audio_key: str, name of column/key containing audio file names 
    :param task_key: str, name of column/key containing tasks
    :param audio_dir: pathlike, path to directory containing audio files. A json or csv with audio metadata should exist in this directory (default = None)
    :param split_dir: pathlike, path to directory to save splits to. May have existing splits. (default = None)
    :param proportions: list of float, list with proportions for each split (default = [.7, .15, .15])
    :param seed: int, random seed for splitting consistently (default = 42)
    :param save: bool, specify whether to save created split (default = False)
    :param load_existing: bool, specify whether to load existing split (default = False)
    :param as_json: bool, specify whether loading/saving should use .json (default = False)
    :param target_tasks: List of target tasks to keep in split (default = None)
    :param target_features: List of target features to stratify on/keep (default = None)
    :param stratify_threshold: int, specify threshold for stratification of features (default = 10)
    :param bucket: GCS bucket (default = None)
    :return train_df: pd.DataFrame, train split
    :return val_df: pd.DataFrame, validation split
    :return test_df: pd.DataFrame, test split
    """
        
    pd.set_option("future.no_silent_downcasting", True)
    #CHECk PROPORTIONS
    assert len(proportions) == 3, 'Must give a proportion for train/val/test even if validation or test set is 0.'
    assert sum(proportions) == 1, 'Proportions must add up to 1'

    prop_inds = np.argwhere(np.array(proportions) != 0)
    prop_inds = [i[0] for i in prop_inds.tolist()]

    #CHECK DIRECTORIES
    assert (audio_dir is not None) or (split_dir is not None), 'At least one of audio_dir or split_dir must be given.'

    if audio_dir is not None: 
        if bucket is None:
            if not isinstance(audio_dir, Path): audio_dir = Path(audio_dir)
            assert audio_dir.exists(), 'Given audio_dir is not an existing directory.'
        else:
            if not isinstance(audio_dir, str): audio_dir = str(audio_dir)
            existing = search_gcs('*', audio_dir, bucket)
            assert existing != [], 'Given audio_dir is not an existing directory.'

    if split_dir is not None:
        if bucket is None:
            if not isinstance(split_dir, Path): split_dir = Path(split_dir)
        else:
            if not isinstance(split_dir, str): split_dir = str(split_dir)

    if (audio_dir is None) and (split_dir is not None):
        load_existing = True
    elif (audio_dir is not None) and (split_dir is None): #only set audio dir to split dir if creating a new split
        load_existing = False 
        print('load_existing set to False as split dir was not given.')
        split_dir = audio_dir

    # SPLIT NAME
    if audio_dir is None:
        name = Path(split_dir).name #assumes you have given full file path
    else:
        name = f'{Path(audio_dir).name}_seed{seed}_tr{proportions[0]}v{proportions[1]}te{proportions[2]}'
        if target_tasks is not None:
            name += f'_ntasks{len(target_tasks)}'
        if target_features is not None: 
            name += f'_nfeats{len(target_features)}'

        if bucket:
            split_dir = f'{split_dir}/{name}'
        else:
            split_dir = split_dir / name 

    
    if load_existing and (split_dir is not None):
        if bucket is None:
            if not split_dir.exists():
                assert audio_dir is not None, 'Cannot load from split_dir as it does not exist. Audio_dir not given, so split cannot be created either.'
                load_existing = False
                print('load_existing set to False as split dir not yet created.')
        else:
            existing = search_gcs('*', split_dir, bucket)
            if existing == []:
                assert audio_dir is not None, 'Cannot load from split_dir as it does not exist. Audio_dir not given, so split cannot be created either.'
                load_existing = False
                print('load_existing set to False as split dir not yet created.')                         
    
    # Load existing
    if load_existing:
        train_df, val_df, test_df = _load_existing(split_dir, as_json, date_key, bucket)
        if (train_df is not None) or (test_df is not None) or (val_df is not None):
            return train_df, val_df, test_df
        else:
            assert audio_dir is not None, 'Split dir has no existing split files. Audio_dir must be given to create them.'

    # create new split
    if as_json: 
        pattern = '.json'
    else:
        pattern = '.csv'

    if bucket:
        paths = search_gcs(pattern, audio_dir, bucket)
    else:
        paths = [p for p in audio_dir.rglob(f'*.{pattern}')]
    
    if paths == []:
        raise ValueError('There must be a metadata file in the audio directory. Please confirm that the metadata file exists, is either a csv or json, and does not have a name of `train`, `val`, or `test`.')
    elif len(paths) > 1:
        raise ValueError('There can be at most one metadata file in audio_dir. Please remove any extra csvs/jsons (depending on as_json) in the audio directory.')
    
    metadata_df = _load(paths[0], bucket, as_json)

    assert date_key in list(metadata_df.columns), 'Date key does not exist in metadata table.'
    assert subject_key in list(metadata_df.columns), 'Subject key does not exist in metadata table.'
    assert audio_key in list(metadata_df.columns), 'Audio key does not exist in metadata table.'


    if target_tasks is not None:
        assert task_key in list(metadata_df.columns), 'Task key must be in metadata if working with target tasks.'
        assert all([t in _TASKS for t in target_tasks]), f'One of {target_tasks} is not valid. Assert that all tasks are in {_TASKS}.'
        metadata_df = metadata_df.loc[metadata_df[task_key].isin(target_tasks)].copy() #happens on original table as we only care about the target tasks

    if target_features is not None:
        assert all([f in _FEATURES for f in target_features]), f'At least one target features is not valid. Assert that all features are in {_FEATURES}.'
        assert all([f in list(metadata_df.columns) for f in target_features]), 'Some target features are not in the metadata table.'
    else:
        target_features = [c for c in metadata_df.columns if c in _FEATURES]

    metadata_df[date_key] = pd.to_datetime(metadata_df[date_key])

    table = metadata_df.reset_index(drop=True).copy() #make a copy of the table to work on 
    table = table.sort_values([subject_key, date_key]).drop_duplicates(subset=subject_key, keep='first').reset_index(drop=True).copy()
    # binarize the columns
    possible_cols = [c for c in table.columns if c in _FEATURES]

    table = table[[subject_key]+possible_cols].copy()
    table[possible_cols] = table[possible_cols].replace([-1.0, 1.0, 1.5,2.0,3.0,4.0,5.0,6.0], [float("nan"), 0.0, 0.0, 1.0, 1.0,1.0,1.0, float("nan")])
    table = table.dropna().copy()

    table['stratify']=table[target_features].apply(lambda x: x.astype(str).str.cat(sep='_'), axis=1)
    group_counts = table[[subject_key,'stratify']].groupby('stratify').aggregate('count').reset_index()
    groups_to_merge = group_counts[group_counts[subject_key]<stratify_threshold].stratify.values
    table['stratify']=np.where(table['stratify'].isin(groups_to_merge), 'merged', table['stratify'])

    #OPTION 1, only train/only val/only test
    if len(prop_inds) == 1:
        if prop_inds[0] == 0:
            train_df = table.copy()
            train_subjects = table[subject_key]
            val_df = None
            val_subjects = []
            test_df = None
            test_subjects = []
        elif prop_inds[0] == 1:
            train_df = None
            train_subjects=[]
            val_df = table.copy()
            val_subjects = table[subject_key]
            test_df = None
            test_subjects = []
        else:
            train_df = None
            train_subjects = []
            val_df = None
            val_subjects = []
            test_df = table.copy()
            test_subjects = table[subject_key]

    #OPTION 2, only train/test OR train/val/test (test subjects needed)
    elif len(prop_inds) == 2 and (prop_inds[0] == 0 and prop_inds[1] == 2):
        _, test_subjects = _sklearn_split(table, subject_key, proportions[prop_inds[-1]], seed)
        train_subjects = table[~table[subject_key].isin(test_subjects)][subject_key].values
        val_subjects = []

    elif len(prop_inds) == 2 and (prop_inds[0] == 0 and prop_inds[1] == 1):
        test_subjects = []
        train_val_df = table.copy()
        train_subjects, val_subjects = _sklearn_split(train_val_df, subject_key,proportions[prop_inds[-1]], seed)

    elif len(prop_inds) == 3:
        _ , test_subjects = _sklearn_split(table, subject_key, proportions[prop_inds[-1]],seed)
        train_val_df = table[~table[subject_key].isin(test_subjects)].copy()
        # rescale val as proportions of trainval
        val_rescaled = proportions[1]/(1-proportions[2])
        train_subjects, val_subjects = _sklearn_split(train_val_df, subject_key, val_rescaled, seed)
        
    else:
        raise ValueError('Invalid combination of proportions.')

    print(f'split results:\n train size = {len(train_subjects)} speakers\n val size = {len(val_subjects)} speakers\n test size = {len(test_subjects)} speakers')

    table = metadata_df[[subject_key, date_key, task_key, audio_key]+target_features].join(metadata_df[target_features].add_prefix('rank_')).copy()
    rank_cols = ['rank_'+c for c in target_features].copy()
    table_binary = table[[subject_key,date_key, task_key, audio_key]+target_features].copy()
    table_binary[target_features] = table_binary[target_features].replace([-1.0, 1.0, 1.5,2.0,3.0,4.0,5.0,6.0], [float("nan"), 0.0, 0.0, 1.0, 1.0,1.0,1.0, float("nan")])
    table_binary = table_binary.dropna().copy()
    table_rank =  table[[audio_key]+rank_cols].copy()
    table_rank[rank_cols] = table_rank[rank_cols].replace([-1.0,1.0, 1.5, 6.0], [float("nan"), 0.0, 1.0, float("nan")])
    table_rank = table_rank.dropna().copy()
    table = table_binary.merge(table_rank).copy()

    if list(train_subjects) != []:
        train_df = table[table[subject_key].isin(train_subjects)].sort_values([subject_key,date_key]).drop_duplicates().reset_index(drop=True).copy()
        train_df[date_key] = train_df[date_key].dt.strftime('%Y-%m-%d')
    else:
        train_df = None 
    
    if list(val_subjects) != []:
        val_df = table[table[subject_key].isin(val_subjects)].sort_values([subject_key,date_key]).drop_duplicates().reset_index(drop=True).copy()
        val_df[date_key] = val_df[date_key].dt.strftime('%Y-%m-%d')
    else:
        val_df = None 
    
    if list(test_subjects) != []:
        test_df = table[table[subject_key].isin(test_subjects)].sort_values([subject_key,date_key]).drop_duplicates().reset_index(drop=True).copy()
        test_df[date_key] = test_df[date_key].dt.strftime('%Y-%m-%d')
    else:
        test_df = None
    
    temp_dict = {'train': train_df, 'val': val_df, 'test': test_df}
    
    if save:
        if bucket is None:
            if split_dir.name != name:
                split_dir = split_dir / name
            if split_dir.exists():
                print('Potentially overwriting existing splits.')
            else:
                split_dir.mkdir(parents=True, exist_ok=True)

            if as_json:
                store = {}
                for t in temp_dict:
                    if temp_dict[t] is not None:
                        store[t] = temp_dict[t].to_dict()
            
                for k in store.keys():
                    p = str(split_dir / (k+'.json'))
                    with open(p, 'w') as file:
                        json.dump(store[k], file, indent=4)
            else:
                for t in temp_dict:
                    if temp_dict[t] is not None:
                        temp_dict[t].to_csv(str(split_dir / (t + '.csv')), index=False)
        else:
            if Path(split_dir).name != name:
                split_dir = Path(split_dir) / name
            existing = search_gcs('*', str(split_dir), bucket)
            if existing != []:
                print('Potentially overwriting existing splits.')
            if as_json:
                store = {}
                for t in temp_dict:
                    if temp_dict[t] is not None:
                        store[t] = temp_dict[t].to_dict()
            
                for k in store.keys():
                    p = f'./{k}.json'
                    with open(p, 'w') as file:
                        json.dump(store[k], file, indent=4)
                    upload_to_gcs(str(split_dir), path = p, bucket=bucket)
                    os.remove(p)
            else:
                for t in temp_dict:
                    if temp_dict[t] is not None:
                        p = f'./{t}.csv'
                        temp_dict[t].to_csv(p, index=False)
                        upload_to_gcs(str(split_dir), path = p, bucket=bucket)
                        os.remove(p)

    return train_df, val_df, test_df