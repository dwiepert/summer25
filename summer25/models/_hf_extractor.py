"""
Hugging Face feature extractor

Author(s): Daniela Wiepert
Last modified: 07/2025
"""
#IMPORT
##built-in
import os
from pathlib import Path
import shutil
from typing import Union, Optional, Tuple

##third-party
from huggingface_hub import snapshot_download
import torch
from transformers import AutoFeatureExtractor, WhisperFeatureExtractor

##local
from summer25.constants import _MODELS
from summer25.io import search_gcs, download_to_local
from ._base_extractor import BaseExtractor

class HFExtractor(BaseExtractor):
    """
    Base extractor

    :param model_type: str, type of model being initialized
    :param pt_ckpt: pathlike, path to base pretrained model checkpoint (default=None)
    :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt
    :param delete_download: bool, specify whether to delete any local downloads from hugging face (default = False)
    :param normalize: bool, specify whether to normalize audio
    :param bucket: gcs bucket
    :param test_hub_fail: bool, TESTING ONLY
    :param test_local_fail: bool, TESTING ONLY
    """
    def __init__(self, model_type:str, pt_ckpt:Optional[Union[Path,str]]=None, from_hub:bool=True, delete_download:bool=False, normalize:bool=False, 
                bucket=None, test_hub_fail:bool=False, test_local_fail:bool=False):
        
        super().__init__(model_type, pt_ckpt)
        self.bucket = bucket
        assert 'hf_hub' in _MODELS[self.model_type], f'{self.model_type} is incompatible with HFModel class.'

        if _MODELS[self.model_type]['use_featext']:
            self.hf_hub = _MODELS[self.model_type]['hf_hub']
            self.from_hub = from_hub
            self.delete_download = delete_download
            self.normalize = normalize

            if not self.from_hub: 
                assert self.pt_ckpt is not None, 'Must give pt_ckpt if not loading from the hub'
                if self.bucket:
                    existing = search_gcs(self.pt_ckpt, self.pt_ckpt, self.bucket)
                    assert existing != [], 'Given pt_ckpt does not exist.'
                    all([e != self.pt_ckpt for e in existing]), 'Expects a directory for hugging face model checkpoints'
                else:
                    if not isinstance(self.pt_ckpt, Path): self.pt_ckpt = Path(self.pt_ckpt)
                    assert self.pt_ckpt.exists(), 'Given pt_ckpt does not exist.'
                    assert self.pt_ckpt.is_dir(), 'Expects a directory for hugging face model checkpoints'
            
            #if self.pt_ckpt is not None: #hugging face models don't require pt ckpt but could be used as a backup
            #    if not isinstance(self.pt_ckpt,str): self.pt_ckpt = str(self.pt_ckpt)
            
            self._load_extractor(test_hub_fail=test_hub_fail, test_local_fail=test_local_fail)
            self._set_kwargs()

            if (self.bucket or self.from_hub) and self.delete_download:
                shutil.rmtree(str(self.local_path))

                bp = Path('.').absolute()
                curr_parent = Path(self.local_path).parent
                while bp.name != curr_parent.name: 
                    os.rmdir(curr_parent)
                    temp = curr_parent.parent 
                    curr_parent = temp
        else:
            self.feature_extractor=None
            self.local_path = None
    
    ### private helpers ###
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
                    return 

                except: 
                    assert self.pt_ckpt is not None, 'Downloading from hub failed, but backup pt_ckpt not available.'
                    print('Downloading from hub failed. Trying pt_ckpt.')

        try:
            if self.bucket:
                local_path = Path('.')
                files = download_to_local(self.pt_ckpt, local_path, self.bucket, directory=True)
                self.pt_ckpt = files[0].parents[0].absolute()
                print(f'{self.pt_ckpt} exists = {self.pt_ckpt.exists()}')
                
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(str(self.pt_ckpt), trust_remote_code=True)
            self.local_path = self.pt_ckpt

        except:
            if (self.bucket or self.from_hub) and self.delete_download:
                shutil.rmtree(str(self.local_path))

                bp = Path('.').absolute()
                curr_parent = Path(self.local_path).parent
                while bp.name != curr_parent.name: 
                    os.rmdir(curr_parent)
                    temp = curr_parent.parent 
                    curr_parent = temp
            raise ValueError('Pretrained checkpoint is incompatible with HuggingFace models. Confirm this is a path to a local hugging face checkpoint.')
    
    def _set_kwargs(self):
        """
        Set kwargs for feature extractor
        """
        self.feature_extractor_kwargs = {}
        self.feature_extractor_kwargs['return_attention_mask'] = True
        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            self.features_key = 'input_features'
            self.attention_key = 'attention_mask'
        else:
            self.features_key = 'input_values'
            self.attention_key = 'attention_mask'
            self.feature_extractor_kwargs['do_normalize'] = self.normalize
            self.feature_extractor_kwargs['padding'] = True
    
    ### main function(s) ###
    def __call__(self, sample:dict) -> dict:
        """
        Run feature extraction on a sample
        :param sample: dict
        :return new_sample: dict
        """
        new_sample = sample.copy()
        wav = new_sample['waveform']

        if self.feature_extractor:
            wav = [torch.squeeze(w).numpy() for w in wav]
            preprocessed_wav = self.feature_extractor(wav,
                                                        return_tensors='pt', 
                                                        sampling_rate = self.feature_extractor.sampling_rate,
                                                        **self.feature_extractor_kwargs)
            new_sample['waveform'] = preprocessed_wav[self.features_key]
            new_sample['attn_mask'] = preprocessed_wav[self.attention_key].bool()
            #return preprocessed_wav[self.features_key], preprocessed_wav[self.attention_key].bool()
        else:
            new_sample['attn_mask'] = None
        
        return new_sample