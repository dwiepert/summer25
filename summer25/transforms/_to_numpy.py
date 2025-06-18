"""
Convert waveform to numpy

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##third-party
import numpy as np

class ToNumpy(object):
    """
    Convert waveform to numpy
    """
    def __call__(self, sample:dict) -> dict:
        """
        :param sample:dict, input sample
        :return npsample: dict, sample after converting waveform to numpy
        """
        npsample = sample.copy()
        waveform = npsample['waveform']
        npsample['waveform'] = np.array(waveform)

        return npsample