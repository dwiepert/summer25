"""
Test hugging face models

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
from pathlib import Path
import shutil

##third-party
import pytest

##local
from summer25.models import HFModel, Classifier


@pytest.mark.hf
def test_hfmodel_pretrained_base():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'
    m = HFModel(**params)
    assert m is not None, 'Model not running properly.'

    #invalid model type (not hugging face)
    params['model_type']='test_model'
    with pytest.raises(AssertionError):
        m = HFModel(**params)
    
    #not loading from hub but not given pt ckpt
    params['model_type'] = 'wavlm-base'
    params['from_hub'] = False
    with pytest.raises(AssertionError):
        m = HFModel(**params)
    
    params['from_hub'] = True
    with pytest.raises(AssertionError):
        m = HFModel(test_local_fail=True, **params)

    params['from_hub'] = False
    #ckpt named but doesn't exist
    pt_ckpt = Path('./ckpt')
    if pt_ckpt.exists():
        shutil.rmtree(pt_ckpt)
    params['pt_ckpt'] = pt_ckpt
    with pytest.raises(AssertionError):
        m = HFModel(**params)

    #checkpoint created but has no models
    pt_ckpt.mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        m = HFModel(**params)
    
    #confirm it defaults back to the hub even if given checkpoint (no failure)
    params['from_hub'] = True
    m = HFModel(**params)
    assert m is not None, 'Model not running properly.'

    #test loading local directory
    m = HFModel(test_hub_fail=True, **params) #should work no issue - DON'T DELETE
    assert m is not None, 'Model not running properly.'
    assert m.local_path.exists(), 'Local path with copy of checkpoint does not exist.'
    
    #test exception raised if local failure and incompatible checkpoint
    pt_ckpt = m.local_path
    with pytest.raises(ValueError):
        m = HFModel(test_local_fail=True, **params)

    #check no failure if given a true ckpt and fails
    params['pt_ckpt'] = m.local_path
    m = HFModel(test_local_fail=True, **params)
    assert m is not None, 'Model not running properly.'

    #check no failure if loading directly from checkpoint
    params['from_hub'] = False
    m = HFModel(**params)
    assert m is not None, 'Model not running properly.'

    #check that you can delete a checkpoint download
    params['from_hub'] = True
    params['delete_download'] = True
    del params['pt_ckpt'] 

    m = HFModel(test_hub_fail=True, **params)
    assert not m.local_path.exists(), 'Local path to checkpoint not deleted.'
    shutil.rmtree(params['out_dir'])
    if pt_ckpt.exists():
        shutil.rmtree(pt_ckpt)
    
def test_hfmodel_finetuned_base():
    params = {'out_dir':Path('./out_dir')}

    #base test
    params['model_type'] = 'wavlm-base'

    ### FINETUNED CHECKPOINT
    ft_ckpt = params['out_dir'] / 'weights'
    if ft_ckpt.exists():
        shutil.rmtree(ft_ckpt)
    params['ft_ckpt'] = ft_ckpt

    # doesn't exist
    with pytest.raises(AssertionError):
        m = HFModel(**params)

    #exists but no files
    ft_ckpt.mkdir(parents=True, exist_ok=True)
    with pytest.raises(AssertionError):
        m = HFModel(**params)
    
    del params['ft_ckpt']

    # check SAVING
    m = HFModel(**params)
    m.save_model_components()
    assert (ft_ckpt/(m.model_name + '.pt')).exists(), 'Base model properly saved.'
    assert (ft_ckpt/(m.clf.config['clf_name'] + '.pt')).exists(), 'Classifier properly saved.'

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
    with pytest.raises(AssertionError):
        m = HFModel(**params)
    
    #Incompatible classifier checkpoint but compatible ft_ckpt
    ckpt = Path('./ckpt')
    if ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(exist_ok=True)

    clf_params = {'in_features':1, 'out_features':1}
    m = Classifier(**clf_params)
    m.save_classifier(ckpt)
    out_ckpt = ckpt / 'weights' 
    out_path = out_ckpt / (m.config['clf_name']+'.pt')
    assert out_path.exists(), 'Classifier not saved properly'
    
    #create new model with different params and try to load old ckpt
    params['nlayers'] = 1
    params['clf_ckpt'] = out_path
    params['ft_ckpt'] = model_ft_ckpt
    with pytest.raises(ValueError):
        m = HFModel(**params)

    shutil.rmtree(params['out_dir'])
    if ft_ckpt.exists():
        shutil.rmtree(ft_ckpt)
    if ckpt.exists():
        shutil.rmtree(ckpt)
    #sth w use feat ext?

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

@pytest.mark.hf
def test_load_hfmodels():
    #TODO: check that the base model for each of the possible hugging face models loads in properly with no errors
    pass
