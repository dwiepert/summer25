"""
Load waveform

Author(s): Daniela Wiepert
Last modified: 07/2025
"""
#IMPORTS
##built-in
import io
from typing import Tuple, Union
from pathlib import Path

##third party
import librosa
import torch
import torchaudio
import torch.nn.functional


def load_waveform(input_dir:Union[Path, str], uid:str, extension:str='wav', lib:bool=False, structured:bool=False, bucket=None) -> Tuple[torch.Tensor, int]:
    """
    load audio from google cloud storage
    :param input_dir: pathlike, path leading to audio
    :param uid: str, audio identifier
    :param extension:str, audio type (default = wav)
    :param lib: boolean indicating to load with librosa rather than torchaudio (default = False)
    :param structured: boolean indicating whether to load from a structured directory (prefix/uid/waveform.wav) or not (default = False)
    :param bucket: gcs bucket object

    :return waveform: torch tensor, loaded audio waveform
    :return sr: int, sample rate
    """
    if not isinstance(input_dir, Path): input_dir=Path(input_dir)
    if structured: 
        waveform_path = input_dir / f'{uid}'
        waveform_path = waveform_path / f'waveform.{extension}'
    else:
        waveform_path = input_dir / f'{uid}.{extension}'

    if bucket: 
        blob = bucket.blob(str(waveform_path))
        wave_string = blob.download_as_string()
        wave_input = io.BytesIO(wave_string)
    else:
        wave_input = waveform_path 
    

    if not lib:
        waveform, sr = torchaudio.load(wave_input, format = extension)
    else:
        waveform, sr = librosa.load(wave_input, mono=False, sr=None)
        waveform = torch.from_numpy(waveform)
        if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
        elif waveform.shape[1] == 1 or waveform.shape[1] == 2:
            waveform = torch.transpose(waveform)

    return waveform, sr