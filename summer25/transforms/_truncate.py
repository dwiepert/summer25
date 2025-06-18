"""
Truncate audio

Author(s): NAIP
Last modified: 06/2025
"""
#IMPORTS
##third-party
import torch 

class Truncate(object):
    '''
    Cut audio to specified length with optional offset
    :param length: float, length to trim to in terms of s
    :param offset: float, offset for clipping in terms of s (default = 0)
    '''
    def __init__(self, length:float, offset:float = 0):
        
        self.length = length
        self.offset = offset
        
    def __call__(self, sample:dict) -> dict:
        """
        Truncate audio sample
        :param sample: dict, input sample
        :return clipsample: dict, sample after truncations
        """
        clipsample = sample.copy()
        waveform = clipsample['waveform']
        sr = clipsample['sample_rate']
        frames = int(self.length*sr)
        offset_frames = int(self.offset*sr)

        waveform_offset = waveform[:, offset_frames:]
        n_samples_remaining = waveform_offset.shape[1]
        
        if n_samples_remaining >= frames:
            waveform_trunc = waveform_offset[:, :frames]
        else:
            n_channels = waveform_offset.shape[0]
            n_pad = frames - n_samples_remaining
            channel_means = waveform_offset.mean(axis = 1).unsqueeze(1)
            waveform_trunc = torch.cat([waveform_offset, torch.ones([n_channels, n_pad])*channel_means], dim = 1)
            
        clipsample['waveform'] = waveform_trunc
        
        return clipsample