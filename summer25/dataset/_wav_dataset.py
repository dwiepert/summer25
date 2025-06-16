"""
Wav Dataset

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
import random
from typing import List, Union

#third party
from audiomentations import *
import numpy as np
import pandas as pd
import torch
import torchvision


#local
from summer25.constants import _MODELS
from summer25.transforms.audio import *
from summer25.transforms.common import *
from ._base_dataset import BaseDataset
     
class WavDataset(BaseDataset):
    def __init__(self, annotation_df:Union[pd.DataFrame, np.ndarray],  prefix:str, model_type:str,
                 config:dict,target_labels:str=None, bucket=None, feature_extractor=None):
        '''
        Dataset that manages audio recordings. 

        :param annotation_df: either an np.array of uids or dataframe with uids as index and annotations in the columns. For classification, must give a dataframe
        :param prefix: location of files to download (compatible with gcs)
        :param model_type: type of model this Dataset will be used with (e.g. w2v2, whisper)
        :param config:dictionary with transform parameters (ones not specified in _MODELS)
        :param target_labels: str list of targets to extract from data. Can be none only for 'asr'.
        :param bucket: gcs bucket
        :param feature_extractor: initialized feature extractor
        '''

        super().__init__(annotation_df=annotation_df, target_labels=target_labels, transform=None)

        self.model_type = model_type
        self.config = config
        self.bucket = bucket
        
        self.prefix = prefix

        self.use_librosa = self._check_existence('use_librosa')
        if self.use_librosa is None:
            self.use_librosa = False
         
        self.transforms = self._get_transforms()

        assert self.target_labels is not None, 'Target labels must be given for classification.'
        assert isinstance(self.annotations_df, pd.DataFrame), 'Must give a dataframe of uids and annotations for classification.'

        if feature_extractor is not None:
            self.set_feature_extractor(feature_extractor)
        
    def _check_existence(self, dictionary:dict, key: str, return_dict:bool=False):
        """
        Check if item exists in a dictionary
        :param dictionary: dict, input dictionary
        :param key: str, key to check
        :param return_dict: bool, indicate whether to return a dictionary with the params
        :return: item from config
        """
        if key in dictionary:
            if return_dict:
                return {key: dictionary.get(key)}
            else:
                return dictionary.get(key)
        else:
            return None
    
    def _get_transforms(self):
        """
        Get transforms for Classification task
        :param transforms: initialized transforms
        """
        #CHECK MODEL CONFIG FIRST
        self.resample_rate = self._check_existence(_MODELS,'target_sample_rate')
        self.monochannel = self._check_existence(_MODELS, 'monochannel')
        self.clip_length = self._check_existence(_MODELS, 'clip_length')
        if self.clip_length is None:
            self.clip_length = self._check_existence(self.config, 'clip_length')

        self.trim_level = self._check_existence(self.config,'trim_level')
        self.mixup = self._check_existence('mixup')
        self.to_tensor = True

        #audiomentation options
        self.gauss = self._check_existence(self.config, 'gauss', return_dict=True)
        self.gausssnr = self._check_existence(self.config,'gausssnr', return_dict=True)
        self.alias = self._check_existence(self.config,'alias', return_dict=True)
        self.bandstop = self._check_existence(self.config,'bandstop', return_dict=True)
        self.bitcrush = self._check_existence(self.config,'bitcrush', return_dict=True)
        self.clipd = self._check_existence(self.config,'clipd', return_dict=True)
        self.gain  = self._check_existence(self.config,'gain', return_dict=True)
        self.gaint= self._check_existence(self.config,'gaint', return_dict=True)
        self.mp3= self._check_existence(self.config,'mp3', return_dict=True)
        self.norm= self._check_existence(self.config,'norm', return_dict=True)
        self.pshift= self._check_existence(self.config,'pshift', return_dict=True)
        self.pinversion= self._check_existence(self.config,'pinversion', return_dict=True)
        self.tstretch= self._check_existence(self.config,'tstretch', return_dict=True)
        self.tmask= self._check_existence(self.config,'tmask', return_dict=True)
        self.tanh = self._check_existence(self.config,'tanh', return_dict=True)
        self.repeat = self._check_existence(self.config,'repeat', return_dict=True)
        self.reverse = self._check_existence('self.config,reverse', return_dict=True)
        self.room = self._check_existence(self.config,'room', return_dict=True)
        self.tshift= self._check_existence(self.config,'tshift', return_dict=True)

        self.al_transform = self._audiomentation_options()
        if self.al_transform != []:
            self.al_transform = Compose(self.al_transform)
            self.annotation_df_augmentation = True
        else:
            self.annotation_df_augmentation = False


        transforms = self._getaudiotransforms()

        return transforms

    
    def _getaudiotransforms(self):
        """
        Use audio configuration parameters to initialize classes for audio transformation. 
        Outputs two tranform variables, one for regular audio transformation and one for 
        augmentations using albumentations

        These transformations will always load the audio. 
        :outparam audio_transform: standard transforms
        """

        waveform_loader = UidToWaveform(prefix = self.prefix, bucket=self.bucket, lib=self.use_librosa)
        transform_list = [waveform_loader]
        if self.monochannel:
            #channel_sum = lambda w: torch.sum(w, axis = 0).unsqueeze(0)
            mono_tfm = ToMonophonic()
            transform_list.append(mono_tfm)
        if self.resample_rate is not None: #16000
            downsample_tfm = ResampleAudio(resample_rate=self.resample_rate, librosa=self.use_librosa)
            transform_list.append(downsample_tfm)
        if self.trim_level is not None:
            trim_tfm = TrimBeginningEndSilence(threshold = self.trim_level, use_librosa=self.use_librosa)
            transform_list.append(trim_tfm)
        if self.clip_length is not None: #160000
            truncate_tfm = Truncate(length = self.clip_length)
            transform_list.append(truncate_tfm)
        if self.to_tensor:
            tensor_tfm = ToTensor()
            transform_list.append(tensor_tfm)
        if self.annotation_df_augmentation:
            numpy_tfm = ToNumpy()
            transform_list.append(numpy_tfm)

        transform = torchvision.transforms.Compose(transform_list)

        transforms = [transform]

        if self.al_transform != []:
            transforms.append(self.al_transform)

        return transforms

    def set_feature_extractor(self, feature_extractor):
        """
        Set a feature extractor
        :param feature_extractor: initialized feature extractor
        """
        self.feature_extractor = feature_extractor
    
    def _add_transform(self, transform, p:dict, t_list:list) -> list:
        """
        Add a transform to a list
        :param transform: transform class
        :param p: dictionary for initializing a transform
        :param t_list: current transform list
        :return t: list with appended transform (if transform given)
        """
        t = t_list.copy()
        if p is not None:
            t.append(transform(**p))
        return t
    
    def _audiomentation_options(self) -> list:
        """
        Add audiomentation transforms
        :return t: list of audiomentation transforms
        """
        t = self._add_transform(Shift, self.tshift, [])
        
        t = self._add_transform(RoomSimulator, self.room, t)
        
        t = self._add_transform(Reverse, self.reverse, t)

        t = self._add_transform(RepeatPart, self.repeat, t)

        t = self._add_transform(TanhDistortion, self.tanh, t)

        t = self._add_transform(TimeMask, self.tmask, t)

        t = self._add_transform(TimeStretch, self.tstretch, t)

        t = self._add_transform(PolarityInversion, self.pinversion, t)

        t = self._add_transform(PitchShift, self.pshift, t)

        t = self._add_transform(Normalize, self.norm, t)

        t = self._add_transform(Mp3Compression, self.mp3, t)

        t = self._add_transform(GainTransition, self.gaint, t)

        t = self._add_transform(Gain, self.gain, t)

        t = self._add_transform(ClippingDistortion, self.clipd, t)

        t = self._add_transform(BitCrush, self.bitcrush, t)

        t = self._add_transform(BandStopFilter, self.bandstop, t)

        t = self._add_transform(Aliasing, self.alias, t)

        t = self._add_transform(AddGaussianSNR, self.gausssnr, t)

        t = self._add_transform(AddGaussianNoise, self.gauss, t)

        return t

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
        
        if isinstance(self.annotation_df, pd.DataFrame):
            uid = self.annotation_df.index[idx] #get uid to load
        else:
            uid = self.annotation_df[idx]
    

        if self.target_labels is not None:
            targets = self.annotation_df[self.target_labels].iloc[idx].values #get target labels for given uid
            
        else:
            targets = []
      
        sample = {'uid': uid,
                  'targets':targets}
        
        sample = self.transforms[0](sample)

        if self.annotation_df_augmentation and self.transforms[1] != []:
            sample['waveform'] = self.transforms[1](samples=sample['waveform'], sample_rate = sample['sample_rate']) #audio augmentations
            if not self.to_numpy:
                sample['waveform'] = torch.from_numpy(sample['waveform']).type(torch.float32)

        #TODO: initialize mixup
        if self.mixup is not None:
            mix = Mixup()
            # if self.mixup is None:
            #     sample= mix(sample, None)

            if random.random() < self.mixup: 
                mix_sample_idx = random.randint(0, len(self.annotations_df)-1)
                mix_uid = self.annotations_df.index[mix_sample_idx]
                mix_targets = self.annotations_df[self.target_labels].iloc[mix_sample_idx].values
            
                sample2 = {
                    'uid': mix_uid,
                    'targets': mix_targets
                }
                sample2 = self.self.transforms[0](sample2) #load and perform standard transformation
                if self.annotation_df_augmentation and self.transforms[1] != []:
                    sample['waveform'] = self.transforms[1](samples=sample['waveform'], sample_rate = sample['sample_rate']) #audio augmentations
                    if not self.to_numpy:
                        sample['waveform'] = torch.from_numpy(sample['waveform']).type(torch.float32)

                sample = mix(sample, sample2)
            
            else:
                sample = mix(sample, None)

        if self.feature_extractor is not None:
            sample = self.feature_extractor(sample)
        
        return sample
    
    def __len__(self):
        """
        Return size of dataset
        """
        return len(self.annotation_df)
    
