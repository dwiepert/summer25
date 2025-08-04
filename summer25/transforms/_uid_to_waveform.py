"""
Load waveform from a uid

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
from pathlib import Path
##local
from summer25.io import load_waveform

class UidToWaveform(object):
    '''
    Take a UID, find & load the data, add waveform and sample rate to sample
    :param prefix:str, path prefix for searching
    :param bucket: gcs bucket (default=None)
    :param extension:str, audio file extension (default = None)
    :param lib: bool, indicate whether to load with librosa (default=False, uses torchaudio)
    :param structured: bool, indicate whether audio files are in structured format (prefix/uid/waveform.wav) or not (default=False)
    '''
    
    def __init__(self, prefix:str, bucket=None, extension:str=None, lib:bool=False, structured:bool=False):
    
        self.bucket = bucket
        self.prefix = prefix #either gcs_prefix or input_dir prefix
        if not isinstance(self.prefix, Path): self.prefix = Path(self.prefix)
        self.cache = {}
        self.extension = extension
        self.lib = lib
        self.structured = structured
        
        if self.bucket is None:
            assert self.prefix.exists(), 'Path must exist if loading waveform.'
        
    def __call__(self, sample:dict) -> dict:
        """
        Load waveform
        :param sample: dict, input sample
        :return wavsample: dict, sample after loading
        """
        wavsample = sample.copy()
        uid, = wavsample['uid'],
        cache = {}
        if uid not in self.cache:
            wav, sr = load_waveform(self.prefix, uid, self.extenstion, self.lib, self.structured, self.bucket)
            cache['waveform'] = wav 
            cache['sample_rate'] = sr
            self.cache[uid] = cache
            
        cache = self.cache[uid]
        
        wavsample['waveform'] = cache['waveform']
        wavsample['sample_rate'] = cache['sample_rate']
         
        return wavsample
    
