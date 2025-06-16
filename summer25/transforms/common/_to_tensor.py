"""
Convert labels to torch tensor

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##third-party
import torch

class ToTensor(object):
    '''
    Convert labels to a tensor rather than ndarray
    '''
    def __call__(self, sample:dict) -> dict:
        """
        :param sample: dict, input sample
        :param tensample: dict, sample after torch conversion
        """
        
        tensample = sample.copy()
        tensample['targets'] = torch.from_numpy(tensample['targets']).type(torch.float32)
        
        return tensample