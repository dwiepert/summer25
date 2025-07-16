"""
Test hugging face feature extractor

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
from pathlib import Path
import shutil

##third-party
import pytest

##local
from summer25.models import HFExtractor

@pytest.mark.hf
def test_extractor_pretrained():
    #base test - load from hub
    m = HFExtractor(model_type='wavlm-base')
    assert m is not None, 'Extractor not running properly.'

    #invalid model type (not hugging face)
    with pytest.raises(AssertionError):
        m = HFExtractor(model_type='test_model')

    #not loading from hub but not given pt checkpoint
    # from hub false
    with pytest.raises(AssertionError):
        m = HFExtractor(model_type='wavlm-base', from_hub=False)

    # true but test local fail
    with pytest.raises(AssertionError):
        m = HFExtractor(model_type='wavlm-base', test_local_fail=True)

    #checkpoint named but doesn't exist
    #ckpt named but doesn't exist
    pt_ckpt = Path('./ckpt')
    if pt_ckpt.exists():
        shutil.rmtree(pt_ckpt)
    with pytest.raises(AssertionError):
        m = HFExtractor(model_type='wavlm-base', pt_ckpt=pt_ckpt, from_hub=False)
    #checkpoint created but has no models
    pt_ckpt.mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        m = HFExtractor(model_type='wavlm-base', pt_ckpt=pt_ckpt, from_hub=False)

    #defaults back to hub even if given checkpoint (no failure)
    m = HFExtractor(model_type='wavlm-base', pt_ckpt=pt_ckpt)
    assert m is not None, 'Extractor not running properly.'

    #loading local directory
    m = HFExtractor(model_type='wavlm-base', test_hub_fail=True)
    assert m is not None, 'Extractor not running properly.'
    assert m.local_path.exists(), 'Local path with copy of checkpoint does not exist.'

    #test exception if local failure and incompatible checkpoint
    with pytest.raises(ValueError):
        m = HFExtractor(model_type='wavlm-base', test_local_fail=True, pt_ckpt=pt_ckpt)

    #check no failure if given a true ckpt and fails
    pt_ckpt = m.local_path
    m = HFExtractor(model_type='wavlm-base', test_local_fail=True, pt_ckpt=pt_ckpt)
    assert m is not None, 'Extractor not running properly.'
    
    #check no failure if loading directly from checkpoint
    m = HFExtractor(model_type='wavlm-base', from_hub=False, pt_ckpt=pt_ckpt)
    assert m is not None, 'Extractor not running properly.'
    
    #check delete a checkpoint download
    m = HFExtractor(model_type='wavlm-base', test_hub_fail=True, delete_download=True)
    assert not m.local_path.exists(), 'Local path to checkpoint not deleted.'
    
    if pt_ckpt.exists():
        shutil.rmtree(pt_ckpt)
