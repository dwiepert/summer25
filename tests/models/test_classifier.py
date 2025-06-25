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