"""
Test classifier

Author(s): Daniela Wiepert
Last modified: 08/2025
"""
#IMPORTS
##third-party
import pytest
import torch

##local
from summer25.models import Classifier


##### TESTS #####
def test_classifier_params():
    params = {'in_features':768, 'out_features':1}
    m = Classifier(**params)
    assert m is not None
    
    #invalid classifier type
    params['clf_type'] = 'random'
    with pytest.raises(NotImplementedError):
        m = Classifier(**params)

    #other classifier type
    params['clf_type'] = 'transformer'
    m = Classifier(**params)
    assert m is not None
    del params['clf_type'] 

    #Linear classifier not implemented error for invalid n layers 
    params['nlayers'] = 3
    with pytest.raises(NotImplementedError):
        m = Classifier(**params)
    del params['nlayers']

    #Not implemented error for invalid activation
    params['activation'] = 'random'
    with pytest.raises(NotImplementedError):
        m = Classifier(**params)
    del params['activation']

def test_weight_initialization():
    m1 = Classifier(in_features=768, out_features=2, nlayers=2, seed=100)
    l1 = m1.classifiers[0].classifier.linear0.weight

    #same seed
    m2 = Classifier(in_features=768, out_features=2, nlayers=2, seed=100)
    l2 = m2.classifiers[0].classifier.linear0.weight
    assert torch.equal(l1, l2)

    #different seed
    m3 = Classifier(in_features=768, out_features=2, nlayers=2, seed=42)
    l3 = m3.classifiers[0].classifier.linear0.weight
    assert not torch.equal(l1, l3)
