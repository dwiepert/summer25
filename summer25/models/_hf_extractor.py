"""
Hugging Face feature extractor

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORT
##built-in
import os
from pathlib import Path
import shutil
from typing import Union, Optional,Dict

##third-party
from huggingface_hub import snapshot_download
from transformers import AutoFeatureExtractor, WhisperFeatureExtractor

##local
from summer25.constants import _MODELS
from ._base_extractor import BaseExtractor

class HFExtractor(BaseExtractor):
    """
    Base extractor

    :param model_type: str, type of model being initialized
    :param pt_ckpt: pathlike, path to base pretrained model checkpoint (default=None)
    :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt
    :param test_hub_fail: bool, TESTING ONLY
    :param test_local_fail: bool, TESTING ONLY
    """
    def __init__(self, model_type:str, pt_ckpt:Optional[Union[Path,str]]=None, from_hub:bool=True,
                test_hub_fail:bool=False, test_local_fail:bool=False, **kwargs):
        super().__init__(model_type, pt_ckpt)

        assert 'hf_hub' in _MODELS[self.model_type], f'{self.model_type} is incompatible with HFModel class.'

        if _MODELS[self.model_type]['use_featext']:
            self.hf_hub = _MODELS[self.model_type]['hf_hub']
            if "delete_download" in kwargs:
                self.delete_download = kwargs.pop("delete_download")
            else:
                self.delete_download = False

            self.from_hub = from_hub
            
            if not self.from_hub: 
                assert self.pt_ckpt is not None, 'Must give pt_ckpt if not loading from the hub'
                if not isinstance(self.pt_ckpt, Path): self.pt_ckpt = Path(self.pt_ckpt)
                assert self.pt_ckpt.exists(), 'Given pt_ckpt does not exist.'
            
            #if self.pt_ckpt is not None: #hugging face models don't require pt ckpt but could be used as a backup
            #    if not isinstance(self.pt_ckpt,str): self.pt_ckpt = str(self.pt_ckpt)
            
            self._load_extractor(test_hub_fail=test_hub_fail, test_local_fail=test_local_fail)
            self._set_kwargs()
        else:
            self.feature_extractor=None
            self.local_path = None
        
    def _load_extractor(self, test_hub_fail:bool=False, test_local_fail:bool=False):
        """
        Load hugging face extractor
        
        :param test_hub_fail: bool, for testing purposes to confirm that non-hugging face functionality works (default=False)
        :param test_local_fail: bool, for testing purposes to confirm that failing a local load raises errors (default=False)
        """
        if self.from_hub:
            try: 
                if test_hub_fail or test_local_fail: 
                    raise Exception()
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.hf_hub, trust_remote_code=True)
                self.local_path = None
                return 
            except:
                try:
                    if test_local_fail:
                        raise Exception()
                     
                    print('Loading directly from hugging face hub failed. Downloading model locally...')
                    self.local_path = Path(f'./{self.hf_hub}').absolute()
                    self.local_path.mkdir(parents=True, exist_ok=True)
                    snapshot_download(repo_id=self.hf_hub, local_dir=str(self.local_path))
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(str(self.local_path))
                    if self.delete_download:
                        print('Deleting local copy of checkpoint')
                        shutil.rmtree(str(self.local_path))

                        bp = Path('.').absolute()
                        curr_parent = self.local_path.parent
                        while bp.name != curr_parent.name: 
                            os.rmdir(curr_parent)
                            temp = curr_parent.parent 
                            curr_parent = temp
                    return 

                except: 
                    assert self.pt_ckpt is not None, 'Downloading from hub failed, but backup pt_ckpt not available.'
                    print('Downloading from hub failed. Trying pt_ckpt.')

        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(str(self.pt_ckpt))
            self.local_path = self.pt_ckpt
        except:
            raise ValueError('Pretrained checkpoint is incompatible with HuggingFace models. Confirm this is a path to a local hugging face checkpoint.')
    
    def _set_kwargs(self):
        """
        Set kwargs for feature extractor
        """
        self.feature_extractor_kwargs = {}
        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            self.features_key = 'input_features'
            self.feature_extractor_kwargs['return_attention_mask'] = True
        else:
            self.features_key = 'input_values'
    
    def __call__(self, sample:Dict) -> dict:
        """
        Run feature extraction on a sample
        :param sample: dict, input sample
        :return featsample: dict, sample after running through feature extractor
        """
        if self.feature_extractor:
            featsample = sample.copy()
            wav = featsample['waveform']
            preprocessed_wav = self.feature_extractor(wav.numpy(),
                                                        return_tensors='pt', 
                                                        sampling_rate = self.feature_extractor.sampling_rate,
                                                        **self.feature_extractor_kwargs)
            featsample['waveform'] = preprocessed_wav[self.features_key]
            return featsample
        else:
            return sample