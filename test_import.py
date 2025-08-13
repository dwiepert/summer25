import argparse
import json
from pathlib import Path
from typing import List

##third-party
from google.cloud import storage
from torch.utils.data import DataLoader

##local
from summer25.models import CustomAutoModel
from summer25.dataset import seeded_split, WavDataset, collate_features
from summer25.constants import _MODELS,_FREEZE, _FEATURES, _FINETUNE, _LOSS, _SCHEDULER, _OPTIMIZER
from summer25.training import Trainer
from summer25.io import search_gcs
