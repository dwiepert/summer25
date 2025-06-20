"""
Wav Dataset

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
import random
from typing import List, Union, Any

#third party
from audiomentations import *
import numpy as np
import pandas as pd
import torch
import torchvision


#local
from summer25.constants import _MODELS
from summer25.transforms import *
from ._base_dataset import BaseDataset
     
class WavDataset(BaseDataset):
    def __init__(self, data:pd.DataFrame, prefix:str, model_type:str, uid_col:str,
                 config:dict, target_labels:str, bucket=None, feature_extractor=None, 
                 transforms=None, extension:str='wav', structured:bool=False):
        '''
        Dataset that manages audio recordings. 

        :param data: dataframe with uids as index and annotations in the columns
        :param prefix: location of audio files (compatible with gcs)
        :param model_type: type of model this Dataset will be used with (e.g. w2v2, whisper)
        :param uid_col: str, specify which column is the uid col
        :param config: dictionary with transform parameters (ones not specified in _MODELS)
        :param target_labels: str list of targets to extract from data. Can be none only for 'asr'.
        :param bucket: gcs bucket (default=None)
        :param feature_extractor: initialized feature extractor (default=Nonse)
        :param transforms: torchvision transforms function to run on data (default=None)
        :param extension: str, audio extension
        :param structured: bool, indicate whether audio files are in structured format (prefix/uid/waveform.wav) or not (default=False)
        '''

        super().__init__(data=data, target_labels=target_labels, transforms=None)

        self.model_type = model_type
        self.config = config
        self.bucket = bucket
        self.extension = extension
        self.structured = structured
        self.prefix = prefix

        self.use_librosa = self._check_existence(self.config,'use_librosa')
        if self.use_librosa is None:
            self.use_librosa = False
         
        if transforms is None:
            self._get_audiomentation_transforms()
            self._get_audio_transforms()
        else:
            print('Using given transforms rather than automatically intializing transforms from config.')
            self.transforms = transforms

        self.feature_extractor = None
        if feature_extractor is not None:
            self.set_feature_extractor(feature_extractor)
        
    def _check_existence(self, dictionary:dict, key: str) -> Any:
        """
        Check if item exists in a dictionary
        :param dictionary: dict, input dictionary
        :param key: str, key to check
        :param return_dict: bool, indicate whether to return a dictionary with the params
        :return: item from config either a
        """
        if key in dictionary:
            return dictionary.get(key)
        else:
            return None
    
    def _get_audio_transforms(self):
        """
        Audio transforms intialization
        """
        #CHECK MODEL CONFIG FIRST
        #BASIC TRANSFORMS MUST MATCH MODEL EXPECTATIONS
        self.resample_rate = _MODELS[self.model_type]['target_sample_rate']
        self.monochannel = _MODELS[self.model_type]['monochannel']
        self.clip_length = self._check_existence(_MODELS[self.model_type], 'clip_length')
        self.truncate = self._check_existence(self.config, 'truncate')
        if self.clip_length is not None:
            if self.truncate:
                self.truncate['length'] = self.clip_length
            else:
                self.truncate = {'length':self.clip_length}

        #TRIM SILENCE OPTION
        self.trim_level = self._check_existence(self.config,'trim_level')

        waveform_loader = UidToWaveform(prefix=self.prefix, bucket=self.bucket, extension=self.extension, lib=self.use_librosa, structured=self.structured)
        tensor_tfm = ToTensor()
        transform_list = [waveform_loader, tensor_tfm]
        if self.resample_rate: #16000
            downsample_tfm = ResampleAudio(resample_rate=self.resample_rate)
            transform_list.append(downsample_tfm)
        if self.monochannel:
            #channel_sum = lambda w: torch.sum(w, axis = 0).unsqueeze(0)
            mono_tfm = ToMonophonic()
            transform_list.append(mono_tfm)
        if self.truncate: #160000
            truncate_tfm = Truncate(**self.truncate)
            transform_list.append(truncate_tfm)
        if self.trim_level and self.use_librosa:
            trim_tfm = TrimBeginningEndSilence(threshold = self.trim_level, use_librosa=self.use_librosa)
            transform_list.append(trim_tfm)
        elif self.trim_level:
            trim_tfm = TrimBeginningEndSilence(trigger_level=self.trim_level, use_librosa=self.use_librosa)
            transform_list.append(trim_tfm)

        if self.data_augmentation:
            numpy_tfm = ToNumpy()
            transform_list.append(numpy_tfm)

        self.audio_transforms = torchvision.transforms.Compose(transform_list)


    def set_feature_extractor(self, feature_extractor):
        """
        Set a feature extractor
        :param feature_extractor: initialized feature extractor
        """
        self.feature_extractor = feature_extractor
    
    def audiomentation_options(self):
        return list(self.augmentations.keys())
        
    def _get_audiomentation_transforms(self):
        """
        Add audiomentation transforms
        """
        #audiomentation parameters - DATA AUGMENTATION
        self.augmentations = {'tshift':Shift, 'room':RoomSimulator, 'reverse':Reverse, 'repeat':RepeatPart, 
                  'tanh':TanhDistortion, 'tmask':TimeMask, 'tstretch':TimeStretch, 
                  'pinversion':PolarityInversion, 'pshift':PitchShift, 'norm':Normalize, 
                  'mp3':Mp3Compression, 'gaint':GainTransition, 'gain':Gain, 
                  'clipd':ClippingDistortion, 'bitcrush':BitCrush, 'bandstop':BandStopFilter, 
                  'alias':Aliasing, 'gausssnr':AddGaussianSNR, 'gauss':AddGaussianNoise}
        
        t = []
        for k in self.augmentations:
            aug = self.augmentations[k]
            p = self._check_existence(self.config, k)
            if p:
                t.append(aug(**p))

        if t != []:
            self.al_transforms = Compose(t)
            self.data_augmentation = True
        else:
            self.al_transforms = None
            self.data_augmentation = False

    def __getitem__(self, idx:Union[torch.Tensor, np.ndarray, List[int], int]) -> dict:
        """
        Given an index, load and run transformations then return the sample dictionary

        Will run transformations in this order:
        Standard audio transformations (load audio -> reduce channels -> resample -> clip -> subtract mean) - also convert labels to tensor
        Albumentation transformations (Time shift -> speed tune -> add gauss noise -> pitch shift -> alter gain -> stretch audio)
        Spectrogram transformations (convert to spectrogram -> frequency mask -> time mask -> normalize -> add noise)

        The resulting sample dictionary contains the following info
        'uid': audio identifier
        'waveform': audio (n_channels, n_frames) or audio path
        'fbank': spectrogram (target_length, frequency_bins)
        'sample_rate': current sample rate
        'targets': labels for current file as tensor

        :param idx: index of a sample
        :return sample: dictionary containing sample 
        """
    
        #If not doing mix-up
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        sample = {'uid': self.data.index[idx],
                  'targets':self.data[self.target_labels].iloc[idx].values}
        
        #AUDIO TRANSFORMS
        sample = self.audio_transforms(sample)

        if self.data_augmentation:
            aug_wav = self.al_transforms(samples=sample['waveform'], sample_rate = sample['sample_rate']) #audio augmentations
            sample['waveform'] = torch.from_numpy(aug_wav).type(torch.float32)
       
        if self.feature_extractor is not None:
            sample = self.feature_extractor(sample)
    
        return sample
    
    def __len__(self):
        """
        Return size of dataset
        """
        return len(self.data)
    
