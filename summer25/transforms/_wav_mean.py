"""
Mean subtraction

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
class WaveMean(object):
    '''
    Subtract the mean from the waveform
    '''
    def __call__(self, sample:dict) -> dict:
        """
        :param sample:dict, input sample
        :return meansample: dict, sample after subtracting mean
        """
        meansample = sample.copy()
        waveform = meansample['waveform']
        waveform = waveform - waveform.mean()
        meansample['waveform'] = waveform
        
        return meansample