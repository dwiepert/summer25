"""
Test classifier

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
from pathlib import Path
import shutil

##third-party
import pytest
import torch

##local
from summer25.models import Classifier
from summer25.io import search_gcs, upload_to_gcs, download_to_local

##### HELPERS #####
def load_json():
    with open('./private_loading/gcs.json', 'r') as file:
        data = json.load(file)

    gcs_prefix = data['gcs_prefix']
    bucket_name = data['bucket_name']
    project_name = data['project_name']
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.get_bucket(bucket_name)
    return gcs_prefix, bucket

def remove_gcs_directories(gcs_prefix,bucket, directory='test_split', pattern="*"):
    dir = gcs_prefix + f'{directory}'
    existing = search_gcs(pattern, dir, bucket)
    for e in existing:
        blob = bucket.blob(e)
        blob.delete()
    existing = search_gcs(pattern, dir, bucket)
    assert existing == []
##### TESTS #####
def test_classifier_params():
    params = {'in_features':1, 'out_features':1}
    m = Classifier(**params)
    assert 'ckpt' not in m.get_config(), 'Checkpoint added to config when not given.'

    #Not implemented error for invalid n layers 
    params['nlayers'] = 3
    with pytest.raises(NotImplementedError):
        m = Classifier(**params)
    del params['nlayers']

    #Not implemented error for invalid activation
    params['activation'] = 'random'
    with pytest.raises(NotImplementedError):
        m = Classifier(**params)
    del params['activation']
    #check 

def test_classifier_checkpoints():
    params = {'in_features':1, 'out_features':1}
    #invalid checkpoint 
    ckpt = Path('./ckpt')
    if ckpt.exists():
        shutil.rmtree(ckpt)
    params['ckpt'] = ckpt
    with pytest.raises(AssertionError):
        m = Classifier(**params)
    
    ckpt.mkdir(exist_ok=True)
    with pytest.raises(AssertionError):
        m = Classifier(**params)

    #Test saving 
    del params['ckpt']
    m = Classifier(**params)
    out_ckpt = ckpt / 'weights' 
    m.save_classifier(m.config['clf_name'], out_ckpt)
    out_path = out_ckpt / (m.config['clf_name']+'.pt')
    assert out_path.exists(), 'Classifier not saved properly'

    #load with newly saved ckpt
    params['ckpt'] = out_path
    m = Classifier(**params)
    
    #load with directory
    params['ckpt'] = out_ckpt
    m = Classifier(**params)

    #create new model with different params and try to load old ckpt
    params['nlayers'] = 1
    with pytest.raises(ValueError):
        m = Classifier(**params)

    #Incompatible classifier checkpoint but compatible ft_ckpt
    ckpt = Path('./ckpt')
    if ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(exist_ok=True)

    if ckpt.exists():
        shutil.rmtree(ckpt)

@pytest.mark.gcs
def test_classifier_checkpoints_gcs():
    gcs_prefix, bucket = load_json()
    ckpt_dir = f'{gcs_prefix}checkpoints'
    
    #Test saving w/out delete download
    params = {'in_features':1, 'out_features':1, 'bucket':bucket}
    m = Classifier(**params)
    name = m.config['clf_name']
    m.save_classifier(m.config['clf_name'], ckpt_dir)
    out_path = f'{ckpt_dir}/{name}.pt'
    existing = search_gcs(out_path, out_path, bucket)
    assert existing != [], 'Classifier not saved properly'
    assert (Path('.') / (name+'.pt')).exists() is False, 'Local classifier not saved correctly.'

    #load with newly saved ckpt
    params['ckpt'] = out_path
    m = Classifier(**params)
    assert m.ckpt.exists(), 'Deleted download when not expected'
    
    #load with directory
    params['ckpt'] = ckpt_dir
    params['delete_download'] = True
    m = Classifier(**params)
    assert not m.ckpt.exists(), 'Did not delete download when expected'

    #create new model with different params and try to load old ckpt
    params['nlayers'] = 1
    with pytest.raises(ValueError):
        m = Classifier(**params)
    assert not m.ckpt.exists(), 'Did not delete download when expected'

    #invalid ckpt
    params['ckpt'] = f'{gcs_prefix}other'
    with pytest.raises(AssertionError):
        m = Classifier(**params)


def test_weight_initialization():
    m1 = Classifier(in_features=768, out_features=2, nlayers=2, seed=100)
    l1 = m1.classifier.linear0.weight

        #same seed
    m2 = Classifier(in_features=768, out_features=2, nlayers=2, seed=100)
    l2 = m2.classifier.linear0.weight
    assert torch.equal(l1, l2)

    #different seed
    m3 = Classifier(in_features=768, out_features=2, nlayers=2, seed=42)
    l3 = m3.classifier.linear0.weight
    assert not torch.equal(l1, l3)

def test_transformer_classifier():
    params = {'in_features':768, 'out_features':2, 'nlayers':2, 'num_heads':4, 'layer_type':'transformer'}

    #LOCAL SAVE
    clf = Classifier(**params)
    path = Path('test_clf')
    out_path = path / 'weights'
    out_path.mkdir(parents=True, exist_ok=True)
    name = clf.get_clf_name()
    clf_path = out_path / (name+'.pt')
    if clf_path.exists(): print(f'Overwriting existing classifier head at {str(clf_path)}')
    torch.save(clf.state_dict(), clf_path)
    
    clf2 = Classifier(**params)
    clf2.load_state_dict(torch.load(clf_path, weights_only=True))

    os.remove(clf_ckpt)
    shutil.rmtree(path)

@pytest.mark.gcs
def transformer_classifier_gcs():
    gcs_prefix, bucket = load_json()
    #LOCAL SAVE
    clf = Classifier(**params)
    path = Path('test_clf')
    out_path = path / 'weights'
    out_path.mkdir(parents=True, exist_ok=True)
    name = clf.get_clf_name()
    torch.save(clf.state_dict(), clf_path)
    
    # BUCKET SAVE
    out_path = f'{gcs_prefix}{str(out_path)}'
    existing = search_gcs(f'{name}.pt', str(out_dir), bucket)
    if existing != []: print(f'Overwriting existing classifier head at {str(out_dir)}')
    upload_to_gcs(str(out_path), clf_path, bucket)
    os.remove(clf_path)

    save_path = Path('.')
    files = download_to_local(f'{out_path}/{name}.pt', save_path, bucket)
    clf_ckpt = files[0]
    clf3 = Classifier(**params)
    clf3.load_state_dict(torch.load(clf_ckpt, weights_only=True))
    os.remove(clf_ckpt)
    shutil.rmtree(path)
    remove_gcs_directories(gcs_prefix, bucket, 'test_clf')
