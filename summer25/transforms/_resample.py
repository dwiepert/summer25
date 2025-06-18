"""
Resample transform

Author(s): NAIP
Last modified: 06/2025
"""
#IMPORTS
##third-party
import librosa 
import torchaudio

class ResampleAudio(object):
    '''
    Resample a waveform
    :param resample_rate:int, rate to resample to (default=16000)
    :param librosa: boolean indicating whether to use librosa (default=False)
    '''
    def __init__(self, resample_rate: int = 16000, librosa: bool = False):
        
        self.resample_rate = resample_rate
        self.librosa = librosa
        
    def __call__(self, sample: dict) -> dict:
        """
        Run resampling
        :param sample: dict, sample
        :return resampled: dict, sample after resampling
        """
        resampled = sample.copy()    
        waveform, sample_rate = resampled['waveform'], resampled['sample_rate']
        if sample_rate != self.resample_rate:
            if self.librosa:
                transformed = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self.resample_rate)
            else:
                transformed = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(waveform)
            resampled['waveform'] = transformed
            resampled['sample_rate'] = self.resample_rate
        
        return resampled