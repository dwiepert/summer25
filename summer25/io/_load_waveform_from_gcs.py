#built-in
import io
from typing import Tuple

#third party
import librosa
import torch
import torchaudio
import torch.nn.functional


def load_waveform_from_gcs(bucket, gcs_prefix:str, uid:str, extension:str='wav', lib:bool=False, structured:bool=False) -> Tuple[torch.Tensor, int]:
    """
    load audio from google cloud storage
    :param bucket: gcs bucket object
    :param gcs_prefix: str, prefix leading to object in gcs bucket
    :param uid: str, audio identifier
    :param extension:str, audio type (default = wav)
    :param lib: boolean indicating to load with librosa rather than torchaudio (default = False)
    :param structured: boolean indicating whether to load from a structured directory (prefix/uid/waveform.wav) or not (default = False)

    :return waveform: torch tensor, loaded audio waveform
    :return sr: int, sample rate
    """
    
    if structured:
        gcs_waveform_path = f'{str(gcs_prefix)}/{uid}/waveform.{extension}'
    else:
        gcs_waveform_path = f'{str(gcs_prefix)}/{uid}.{extension}'
    
    blob = bucket.blob(gcs_waveform_path)
    wave_string = blob.download_as_string()
    wave_bytes = io.BytesIO(wave_string)
    if not lib:
        waveform, sr = torchaudio.load(wave_bytes, format = extension)
    else:
        waveform, sr = librosa.load(wave_bytes, mono=False, sr=None)
        waveform = torch.from_numpy(waveform)
        if len(waveform.shape) == 1:
           waveform = waveform.unsqueeze(0)
        elif waveform.shape[1] == 1 or waveform.shape[1] == 2:
            waveform = torch.transpose(waveform)
    
    return waveform, sr