"""
Pad audio

Author(s): NAIP
Last modified: 06/2025
"""
#IMPORTS
##third-party
import torch 

class Pad(object):
    '''
    Pad audio to specified len
    :param pad_method: str, indicate whether to pad with mean or pad with 0
    '''
    def __init__(self, pad_method:str='mean'):
        self.pad_method = pad_method
        assert self.pad_method in ['mean', 'zero'], 'Pad method must be one of `zero` or `mean`'
        
    def __call__(self, sample:dict, max_len:int) -> dict:
        """
        Pad audio sample
        :param sample: dict, input sample
        :return clipsample: dict, sample after truncations
        """
        padsample = sample.copy()
        waveform = padsample['waveform']

        if waveform.shape[1] != max_len:
            n_channels = waveform.shape[0]
            n_pad = max_len - waveform.shape[1]
            if self.pad_method == 'mean':
                channel_means = waveform.mean(axis = 1).unsqueeze(1)
                add = torch.ones([n_channels, n_pad])*channel_means
            else:
                add = torch.zeros([n_channels, n_pad])
            waveform_pad = torch.cat([waveform, add], dim = 1)
        else:
            waveform_pad =  waveform
            n_pad = 0
        padsample['waveform'] = waveform_pad
        padsample['pad_tokens'] = n_pad
        
        return padsample