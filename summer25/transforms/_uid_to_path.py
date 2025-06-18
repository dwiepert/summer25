"""
Get absolute path from a UID

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
import os
from pathlib import Path
from typing import Union

##local
from summer25.io import download_file_to_local

class UidToPath(object):
    """
    Take a UID and convert to an absolute path. Download to local computer if necessary.
    :param prefix:str, path prefix for searching (either local directory or gcs path)
    :param savedir: pathlike, path to directory with objects (default = None)
    :param bucket: gcs bucket (default=None)
    :param ext: str, extension to search for (default=wav)
    :param structured: bool, indicate whether the data is in a structured directory (path/uid/waveform.wav) or not (default=False)
    """

    def __init__(self, prefix:Union[Path, str], savedir:Union[Path,str] = None, bucket=None, ext:str='wav', structured:bool=False):
        self.prefix = prefix
        if not isinstance(self.prefix,Path): self.prefix = Path(self.prefix)
        self.savedir = savedir
        self.bucket = bucket
        self.ext = ext
        self.structured = structured

        if self.bucket is not None:
            assert self.savedir is not None, 'must have a directory to save to if downloading from bucket'
            if not isinstance(self.savedir,Path): self.savedir = Path(self.savedir)
            self.savedir = self.savedir.absolute()
            if not self.savedir.exists():
                os.makedirs(savedir)
        else:
            assert self.prefix.exists(), 'Prefix must exist if not using a bucket.'
        self.cache = {}

    def __call__(self, sample:dict) -> dict:
        """
        Run
        :param sample:dict, input sample
        :return pathsample: dict, sample after running uid to path
        """
        pathsample = sample.copy()
        uid = pathsample['uid']
        
        cache_uid = []
        cache_waveform = []
        if uid not in self.cache:
            if self.structured:
                temp_path = self.prefix / uid
                temp_audio_path = temp_path / f'waveform.{self.ext}'
            else:
                temp_audio_path = self.prefix / f"{uid}.{self.ext}"

            cache = {}

            if self.bucket is None:
                assert temp_audio_path.exists(), 'Path to audio must exist.'
                cache['waveform'] = str(temp_audio_path.absolute())
                self.cache[uid] = cache
        
            else:
                if self.structured:
                    save_path = self.savedir /uid
                    save_path = save_path / f'waveform.{self.ext}'
                else:
                    save_path = self.savedir / f"{uid}.{self.ext}"

                download_file_to_local(temp_audio_path, save_path, self.bucket)
                
                cache['waveform'] = str(download_file_to_local(temp_audio_path, save_path, self.bucket))
                self.cache[uid] = cache

        cache = self.cache[uid]
        pathsample['waveform'] = cache['waveform']


        return pathsample

