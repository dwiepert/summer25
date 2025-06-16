"""
Mix two audio samples together

Author(s): Daniela Wiepert
Last modified: 06/2025
"""

import numpy as np
import torch

class Mixup(object):
    '''
    Implement mixup of two files

    :param sample1: dictionary of first sample
    :param sample2: dictionary of second sample
    :return basesample: sample with waveform updated
    '''

    def __call__(self, sample1:dict, sample2:dict=None):
        basesample = sample1.copy()
        mixsample = sample2.copy()
        if sample2 is None:
            waveform = basesample['waveform']
            waveform = waveform - waveform.mean()
        else:
            waveform1 = basesample['waveform']
            waveform2 = mixsample['waveform']
            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0,0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    waveform2 = waveform2[0,0:waveform1.shape[1]]

            #sample lambda from beta distribution
            mix_lambda = np.random.beta(10,10)

            mix_waveform = mix_lambda*waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()   

            targets1 = basesample['targets']
            targets2 = mixsample['targets']
            targets = mix_lambda*targets1 + (1-mix_lambda)*targets2
            basesample['targets'] = targets
            #TODO: what is happening here

        basesample['waveform'] = waveform

        return basesample
