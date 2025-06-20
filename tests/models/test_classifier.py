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
    m.save_classifier(ckpt)
    out_ckpt = ckpt / 'weights' 
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

    if ckpt.exists():
        shutil.rmtree(ckpt)

def test_weight_initialization():
    #TODO
    pass