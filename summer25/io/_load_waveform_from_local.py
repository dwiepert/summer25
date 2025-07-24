"""
Load waveform from local machine

Author(s): Daniela Wiepert
Last modified: 06/2025
"""

#IMPORTS
##built-in
from pathlib import Path
from typing import Union, Tuple

##third party
import librosa
import torch
import torchaudio

def load_waveform_from_local(input_dir:Union[str,Path], uid:str, extension:str='wav', lib:bool=False, structured:bool=False) -> Tuple[torch.Tensor, int]:
    """
    :param input_directory: pathlike, directory where data is stored locally
    :param uid: audio identifier
    :param extension: audio type (default = wav)
    :param lib: boolean indicating to load with librosa rather than torchaudio (default = False)
    :param structured: boolean indicating whether to load from a structured directory (prefix/uid/waveform.wav) or not (default = False)
    :return waveform: torch tensor, loaded audio waveform
    :return sr: int, sample rate
    """
    if not isinstance(input_dir, Path): input_dir=Path(input_dir)
    if structured:     
        waveform_path = input_dir / f'{uid}'
        waveform_path = waveform_path / f'waveform.{extension}'
    else:
        waveform_path = input_dir / f'{uid}.{extension}'
    
    if not waveform_path.exists():
        raise FileNotFoundError('File does not exist')
    
    if not lib:
        waveform, sr = torchaudio.load(waveform_path, format=extension)
    else:
        waveform, sr = librosa.load(waveform_path, mono=False, sr=None)
        waveform = torch.from_numpy(waveform)
        if len(waveform.shape) == 1:
           waveform = waveform.unsqueeze(0)
        elif waveform.shape[1] == 1 or waveform.shape[1] == 2:
            waveform = torch.transpose(waveform)
    
    return waveform, sr