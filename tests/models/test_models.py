"""
Test models

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
import os
from pathlib import Path
import shutil

##third-party
import pytest

##local
from summer25.models import HFModel, BaseModel, Classifier

def test_basemodel_params():
    params = {'out_dir':Path('./out_dir'), 'pool_dim': 1}
    
    # check model type? add into class HF and base model (2 diff constants)
    params['model_type'] = 'random'
    with pytest.raises(AssertionError):
        m = BaseModel(**params)
    
    params['model_type'] = 'wavlm-base'
    m = BaseModel(**params)

    assert 'pool_dim' in m.base_config and m.base_config['pool_dim'] == params['pool_dim'], 'Pool dim not added to base config'
    assert 'unfreeze_layers' not in m.base_config, 'Unfreeze layers incorrectly added to config'
    assert 'pt_ckpt' not in m.base_config, 'pt_ckpt incorrectly added to base config'
    assert 'ft_ckpt' not in m.base_config, 'ft_ckpt incorrectly added to base config'

    #check freeze method 
    params['freeze_method'] = 'layer'
    with pytest.raises(AssertionError):
        m = BaseModel(**params)
    params['unfreeze_layers'] = 0
    with pytest.raises(AssertionError):
        m = BaseModel(**params)

    params['unfreeze_layers'] = ['encoder.layer0']
    m = BaseModel(**params)
    assert 'unfreeze_layers' in m.base_config and m.base_config['unfreeze_layers'] == params['unfreeze_layers'], 'Unfreeze layers not added to base config.'

    params['unfreeze_layers'] = [0]
    m = BaseModel(**params)
    assert 'unfreeze_layers' in m.base_config and m.base_config['unfreeze_layers'] == params['unfreeze_layers'], 'Unfreeze layers not added to base config.'

    params['unfreeze_layers'] = [{}] #TODO: what would this do?
    with pytest.raises(AssertionError):
        m = BaseModel(**params)

    params['freeze_method'] = 'random'
    with pytest.raises(AssertionError):
        m = BaseModel(**params)
    
    del params['freeze_method']
    del params['unfreeze_layers']

    #check pool method
    params['pool_method'] = 'random'
    with pytest.raises(AssertionError):
        m = BaseModel(**params)
    
    del params['pool_method']

    #no pool dim - mean pooling
    del params['pool_dim']
    with pytest.raises(AssertionError):
        m = BaseModel(**params)

    #pool dim not int or tuple
    params['pool_dim'] = '1'
    with pytest.raises(AssertionError):
        m = BaseModel(**params)

    #pool dim - tuple
    params['pool_dim'] = (0,1)
    m = BaseModel(**params)
    assert 'pool_dim' in m.base_config and m.base_config['pool_dim'] == params['pool_dim'], 'Pool dim not added to base config'
    params['pool_dim'] = ('0','1')
    with pytest.raises(AssertionError):
        m = BaseModel(**params)
    
    #no pool dim + attn 
    params['pool_method'] = 'attn'
    m = BaseModel(**params)
    assert 'pool_dim' not in m.base_config, 'Pool dim incorrectly added to base config'
    params['pool_dim'] = 1
    del params['pool_method']

    # check pretrained ckpt exists
    pt_ckpt = Path('./pt_ckpt')
    if pt_ckpt.exists():
        shutil.rmtree(pt_ckpt)
    params['pt_ckpt'] = pt_ckpt 
    with pytest.raises(AssertionError):
        m = BaseModel(**params)
    pt_ckpt.mkdir(exist_ok=True)
    m = BaseModel(**params)
    assert 'pt_ckpt' in m.base_config and m.base_config['pt_ckpt'] == str(pt_ckpt), 'pt ckpt not added to config correctly.'
    del params['pt_ckpt']
    shutil.rmtree(pt_ckpt)

    # check finetuned ckpt exists
    ft_ckpt = Path('./ft_ckpt')
    if ft_ckpt.exists():
        shutil.rmtree(ft_ckpt)
    params['ft_ckpt'] = ft_ckpt 
    with pytest.raises(AssertionError):
        m = BaseModel(**params)
    
    ft_ckpt.mkdir(exist_ok=True)
    with pytest.raises(AssertionError):
        m = BaseModel(**params)

    del params['ft_ckpt']
    shutil.rmtree(ft_ckpt)
    shutil.rmtree(params['out_dir'])

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

    shutil.rmtree(ckpt)

    #TODO: try forward function of classifier

def test_hfmodel_base():
    #test hf_hub not in model_type (ADD RANDOM TEST MODEL)
    params = {'out_dir':Path('./out_dir')}

    params['model_type']='test_model'
    with pytest.raises(AssertionError):
        m = HFModel(**params)
    
    params['model_type'] = 'wavlm-base'
    params['from_hub'] = False
    with pytest.raises(AssertionError):
        m = HFModel(**params)
    
    pt_ckpt = Path('./ckpt')
    if pt_ckpt.exists():
        shutil.rmtree(pt_ckpt)
    params['pt_ckpt'] = pt_ckpt
    with pytest.raises(AssertionError):
        m = HFModel(**params)

    pt_ckpt.mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        m = HFModel(**params)
    
    params['from_hub'] = True
    m = HFModel(**params)
    """
    m = HFModel(test_hub_fail=True, **params) #should work no issue - DON'T DELETE
    assert m.local_path.exists(), 'Local path with copy of checkpoint does not exist.'
    
    pt_ckpt = m.local_path
    with pytest.raises(AssertionError):
        m = HFModel(test_local_fail=True, **params)

    params['pt_ckpt'] = m.local_path
    m = HFModel(test_local_fail=True, **params)

    params['from_hub'] = False
    m = HFModel(**params)

    params['from_hub'] = True
    params['delete_download'] = True
    del params['pt_ckpt'] 

    m = HFModel(**params)
    assert not m.local_path.exists(), 'Local path to checkpoint not deleted.'
    """
    """
    ### FINETUNED CHECKPOINT
    ft_ckpt = params['out_dir'] / 'weight'
    if ft_ckpt.exists():
        shutil.rmtree(ft_ckpt)
    params['ft_ckpt'] = ft_ckpt

    # doesn't exist
    with pytest.raises(AssertionError):
        m = HFModel(**params)

    #exists but no files
    ft_ckpt.mkdir(parents=True, exist_ok=true)
    with pytest.raises(AssertionError):
        m = HFModel(**params)
    
    del params['ft_ckpt']

    # check SAVING
    m = HFModel(**params)
    m.save_model_components()
    assert ft_ckpt/(m.model_name + '.pt').exists(), 'Base model properly saved.'
    assert ft_ckpt/(m.clf.config['clf_name'] + '.pt').exists(), 'Classifier properly saved.'

    #FT POSSIBILITIES:
    #directly give .pt file and only update base model
    model_ft_ckpt = ft_ckpt/(m.model_name + '.pt')
    params['ft_ckpt'] = model_ft_ckpt
    m = HFModel(**params)
    
    #give a clf_ckpt and update only clf 
    del params['ft_ckpt']
    params['clf_ckpt'] = ft_ckpt/(m.config['clf_name'] + '.pt')
    m = HFModel(**params)

    #give a dir, update only base model (clf_ckpt given)
    params['ft_ckpt'] = ft_ckpt
    m = HFModel(**params)
    del params['clf_ckpt']

    #give a dir, update base model and clf 
    m = HFModel(**params)

    # Incompatible model
    params['model_type'] = 'hubert-base'
    with pytest.raises(ValueError):
        m = HFModel(**params)
    
    shutil.rmtree(params['out_dir'])
    #sth w use feat ext?
"""

def test_freeze():
    #check freeze methods work properly
    #TODO:
    pass

def test_weight_initialization():
    #TODO:
    pass

def test_pooling():
    #TODO:
    #test pooling works properly
    pass

def test_forward():
    #TODO:
    #test forward pass works
    pass


def test_load_hfmodels():
    #TODO: check that the base model for each of the possible hugging face models loads in properly with no errors
    pass

def test_hf_config():
    #TODO:
    #assert 'ft_ckpt' in m.base_config and m.base_config['ft_ckpt'] == str(ft_ckpt), 'ft ckpt not added to config correctly.'
    pass

def test_finetuned():
    #model saves properly
    #load from finetuned ckpt
    pass