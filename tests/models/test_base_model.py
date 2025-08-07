"""
Test base model class 

Author(s): Daniela Wiepert
Last modified: 08/2025
"""
#IMPORTS
from pathlib import Path
import shutil

##third-party
import pytest

##local
from summer25.models import BaseModel

##### TESTS #####
def test_basemodel_params():
    params = {'out_dir':Path('./out_dir'), 'pool_dim': 1}
    
    # check model type? add into class HF and base model (2 diff constants)
    params['model_type'] = 'random'
    with pytest.raises(AssertionError):
        m = BaseModel(**params)
    
    params['model_type'] = 'wavlm-base'
    m = BaseModel(**params)

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
    params['model_type'] = 'test_model2'
    with pytest.raises(ValueError):
        m = BaseModel(**params)

    params['model_type'] = 'wavlm-base'
    #pool dim not int or tuple
    params['pool_dim'] = '1'
    params['pool_method'] = 'mean'
    with pytest.raises(AssertionError):
        m = BaseModel(**params)

    params['pool_dim'] = 'str'
    with pytest.raises(AssertionError):
        m = BaseModel(**params)

    #pool dim - tuple
    params['pool_dim'] = (0,1)
    m = BaseModel(**params)
    params['pool_dim'] = ('0','1')
    with pytest.raises(AssertionError):
        m = BaseModel(**params)
    
    #no pool dim (uses from _MODELS) + attn 
    del params['pool_dim']
    params['pool_method'] = 'attention'
    m = BaseModel(**params)
    #assert 'pool_dim' not in m.base_config, 'Pool dim incorrectly added to base config'
    params['pool_dim'] = 1
    del params['pool_method']

    #check finetune method
    params['finetune_method'] = 'random'
    with pytest.raises(AssertionError):
        m = BaseModel(**params)
    shutil.rmtree(params['out_dir'])