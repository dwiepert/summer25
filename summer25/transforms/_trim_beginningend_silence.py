"""
Trim beginning and end silence

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##third-party
import librosa
import torchaudio.functional as taf
import torchaudio.sox_effects as tase
import torch

class TrimBeginningEndSilence:
    """
    Trim beginning and end silence with librosa. Only use if loaded with librosa.
    :param threshold: int, the db threshold (librosa) (default=60)
    :param use_librosa: bool, toggle for librosa trimming
    :param trigger_level: trigger level (torchaudio) (default=7.0)
    """
    def __init__(self, threshold: int = 60, use_librosa:bool=True, trigger_level:float=7.0):
        self.threshold = threshold
        self.use_librosa = use_librosa
        self.trigger_level = trigger_level
    
    def __call__(self, sample:dict)->dict:
        """
        Run trimming
        :param sample: dict, input sample
        :return trimsample: dict, sample after trimming silence
        """
        trimsample = sample.copy()
        waveform = trimsample['waveform']

        if self.use_librosa:
            out_waveform, _ = librosa.effects.trim(waveform, top_db=self.threshold) #there are some others to consider too, see: https://librosa.org/doc/main/generated/librosa.effects.trim.html
        else:
            sr = trimsample['sample_rate']
            beg_trim = taf.vad(waveform, sample_rate=sr, trigger_level=self.trigger_level)
            if beg_trim.nelement() == 0:
                print('Waveform may be empty. Currently skipping trimming. See TrimBeginningEndSilence for more information.')
                out_waveform = waveform 
            else:
                rev = torch.flip(beg_trim, [0,1])
                #rev, sr = tase.apply_effects_tensor(beg_trim, sample_rate=sr, effects=[['reverse']])
                
                end_trim = taf.vad(rev, sample_rate=sr, trigger_level=self.trigger_level)
                if end_trim.nelement() == 0:
                    print('Waveform may be empty. Currently ignoring. See TrimBeginningEndSilence for more information.')
                    out_waveform = beg_trim
                else:
                    out_waveform = torch.flip(end_trim, [0,1])#tase.apply_effects_tensor(end_trim, sample_rate=sr, effects=[['reverse']])
                
        trimsample['waveform'] = out_waveform

        return trimsample