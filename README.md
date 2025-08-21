# Summer 2025 Project

Train a variety of hugging face transformers to predict pathological speech features. 

## Training a model 
A full example of training exists in [run.py](https://github.com/dwiepert/summer25/run.py).

This package trains with three custom components: [WavDataset](https://github.com/dwiepert/summer25/summer25/dataset/_wav_dataset.py), [CustomAutoModel](https://github.com/dwiepert/summer25/summer25/models/_auto_model.py), and [Trainer](https://github.com/dwiepert/summer25/summer25/training/_trainer.py).

It also contains code to create dataset splits [seeded_split](https://github.com/dwiepert/summer25/summer25/dataset/_split.py) and collate data into a batch [collate_features](https://github.com/dwiepert/summer25/summer25/dataset/_custom_collate.py). 

### Data Loading
This package expects the data to be in one of the following formats:
* structured = False

    `<AUDIO_DIR>/<WAV_ID>.<EXT>`

* structured = True

    `<AUDIO_DIR>/<WAV_ID>/waveform.<EXT>`

A metadata `.csv` file should also exist in the `<AUDIO_DIR>` that contains `<WAV_ID>` values in a specified `audio_key` column which is passed to [seeded_split](https://github.com/dwiepert/summer25/summer25/dataset/_split.py) as a parameter for creating dataset splits.

```
seeded_split(subject_key:str, date_key:str, audio_key:str, task_key:str, audio_dir:Union[Path, str]=None, split_dir:Union[Path,str]=None, proportions:List[float]=[.7, .15, .15], seed:int=42, save:bool=False, load_existing:bool=False, as_json:bool=False, target_tasks:List[str]=None, target_features:List[str] = None, stratify_threshold:int=10, bucket=None)

Create train/test/val splits

    :param subject_key: str, name of column/key containing subject identifiers
    :param date_key: str, name of column/key containing date
    :param audio_key: str, name of column/key containing audio file names 
    :param task_key: str, name of column/key containing tasks
    :param audio_dir: pathlike, path to directory containing audio files. A json or csv with audio metadata should exist in this directory (default = None)
    :param split_dir: pathlike, path to directory to save splits to. May have existing splits. (default = None)
    :param proportions: list of float, list with proportions for each split (default = [.7, .15, .15])
    :param seed: int, random seed for splitting consistently (default = 42)
    :param save: bool, specify whether to save created split (default = False)
    :param load_existing: bool, specify whether to load existing split (default = False)
    :param as_json: bool, specify whether loading/saving should use .json (default = False)
    :param target_tasks: List of target tasks to keep in split (default = None)
    :param target_features: List of target features to stratify on/keep (default = None)
    :param stratify_threshold: int, specify threshold for stratification of features (default = 10)
    :param bucket: GCS bucket (default = None)
    :return train_df: pd.DataFrame, train split
    :return val_df: pd.DataFrame, validation split
    :return test_df: pd.DataFrame, test split
```
    

The data is load with [WavDataset](https://github.com/dwiepert/summer25/summer25/dataset/_wav_dataset.py). 

```
WavDataset(data:pd.DataFrame, prefix:str, model_type:str uid_col:str, config:dict, target_labels:str, rank_prefix:str=None, bucket=None, transforms=None, extensionstr='wav', structured:bool=False)

    :param data: dataframe with uids as index and annotations in the columns
    :param prefix: location of audio files (compatible with gcs)
    :param model_type: type of model this Dataset will be used with (e.g. w2v2, whisper)
    :param uid_col: str, specify which column is the uid col
    :param config: dictionary with transform parameters (ones not specified in _MODELS.
    :param target_labels: str list of targets to extract from data. Can be none only for 'asr'.
    :param rank_prefix: str, prefix for columns with rank target
    :param bucket: gcs bucket (default=None)
    :param transforms: torchvision transforms function to run on data (default=None)
    :param extension: str, audio extension
    :param structured: bool, indicate whether audio files are in structured format (prefix/uid/waveform.wav) or not (default=False)
```
See example of a config for adding additional data augmentations [transform_config.json](TODO). The loaded dictionary is passed as the `config` parameter in `WavDataset`.

#### Data loading example:
```
from torch.utils.data import DataLoader
from summer25.dataset import seeded_split, WavDataset, collate_features 

#generate data split
train_df, val_df, test_df = seeded_split(**split_params)

#initialize dataset
train_dataset = WavDataset(data=train_df, **dataset_params)
val_dataset = WavDataset(data=val_df, **dataset_params)
test_dataset = WavDataset(data=test_df, **dataset_params)

#set up DataLoader with custom collate
train_loader = DataLoader(dataset=train_dataset,batch_size=1,shuffle=True,collate_fn=collate_features, num_workers=0)
val_loader = DataLoader(dataset=val_dataset,batch_size=1,shuffle=False,collate_fn=collate_features, num_workers=0)
test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,collate_fn=collate_features, num_workers=s)
```


### Model initialization 
The basic model is initialized with the `from_pretrained` function of [CustomAutoModel](https://github.com/dwiepert/summer25/summer25/models/_auto_model.py). 

```
CustomAutoModel.from_pretrained(config:dict, ft_checkpoint:Optional[Union[str, Path]]=None, clf_checkpoint:Optional[Union[str,Path]]=None, pt_checkpoint:Optional[Union[str,Path]]=None, delete_download:bool=False, data_parallel:bool=False)

Load a model from a pretrained checkpoint
    :param config: dict, config of model arguments
    :param ft_checkpoint: pathlike, finetuned checkpoint path as a directory (default = None)
    :param clf_checkpoint: pathlike, classifier checkpoint path as a file (default = None)
    :param pt_checkpoint: pathlike, pretrained checkpoint path as a directorr (default = None)
    :param delete_download: bool, specify whether to delete any local downloads from hugging face (default = False)
    :param data_parallel: bool, true if using multiple gpus
    :return model: loaded model
```

In this function, the following conditions must be met:

1. All model arguments that you wish to specify from [HFModel](https://github.com/dwiepert/summer25/summer25/models/_hf_model.py) must be passed as a dictionary (`config`). 

```
HFModel parameters
    :param out_dir: Pathlike, output directory to save to
    :param model_type: str, hugging face model type for naming purposes
    :param finetune_method: str, specify finetune method (default=None)
    :param freeze_method: str, freeze method for base pretrained model (default=required-only)
    :param unfreeze_layers: List[str], optionally give list of layers to unfreeze (default = None)
    :param pool_method: str, pooling method for base model output (default=mean)
    :param normalize: bool, specify whether to normalize input
    :param out_features: int, number of output features from classifier (number of classes) (default = 1)
    :param nlayers: int, number of layers in classification head (default = 2)
    :param activation: str, activation function to use in classification head (default = 'relu')
    :param bottleneck: int, optional bottleneck parameter (default=None)
    :param layernorm: bool, true for adding layer norm (default=False)
    :param dropout: float, dropout level (default = 0.0)
    :param binary:bool, specify whether output is making binary decisions (default=True)
    :param clf_type:str, specify layer type ['linear','transformer'] (default='linear')
    :param num_heads:int, number of encoder heads in using transformer build (default = 4)
    :param separate:bool, true if each feature gets a separate classifier head
    :param lora_rank: int, optional value when using LoRA - set rank (default = 8)
    :param lora_alpha: int, optional value when using LoRA - set alpha (default = 16)
    :param lora_dropout: float, optional value when using LoRA - set dropout (default = 0.0)
    :param virtual_tokens: int, optional value when using soft prompting - set num tokens (default = 4)
    :param seed: int, specify random seed for ensuring reproducibility (default = 42)
    :param device: torch device (default = cuda)
    :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt (default = True)
    :param print_memory: bool, true if printing memory information
    :param bucket: gcs bucket (default = None)
```

2. You must always specify either from_hub or pt_checkpoint. The feature extractor and PEFT models always require a pretrained checkpoint. 

### Training 
The model can be finetuned using [Trainer](https://github.com/dwiepert/summer25/summer25/training/_trainer.py). 

```
Trainer( model:Union[HFModel], target_features:List[str], optim_type:str="adamw", 
                 tf_learning_rate:float=None, learning_rate:float=1e-4, loss_type:str="bce", scheduler_type:str=None,
                 early_stop:bool=False, save_checkpoints:bool=True, patience:int=5, delta:float=0.0, **kwargs)

Custom model training class

    :param model: initialized model
    :param target_features: list of target features
    :param optim_type: str, optimizer type (default = adamw)
    :param learning_rate: float, learning rate (default = 1e-3)
    :param loss_type: str, loss type (default = bce)
    :param scheduler_type: str, scheduler type (default = None)
    :param early_stop: bool, specify whether to use early stopping (default = False)
    :param save_checkpoints: bool, specify whether to save checkpoints (default = True)
    :param patience: int, patience for early stopping (default = 5)
    :param delta: float, minimum change for early stopping (default = 0.0)
    :param kwargs: additional values for rank classification loss or schedulers (e.g., rating_threshold/margin/bce_weight for rank loss and end_lr/epochs for Exponential scheduler)
```

Use the `fit` function to finetune and the `test` function to evaluate the model. 

```
Trainer.fit(train_loader:DataLoader, val_loader:DataLoader=None, epochs:int=10)

Trainer.test(test_loader:DataLoader)
```

### Combined Example
```
from torch.utils.data import DataLoader
from summer25.dataset import seeded_split, WavDataset, collate_features 
from summer25.models import CustomAutoModel
from summer25.training import Trainer

#generate data split
train_df, val_df, test_df = seeded_split(**split_params)

#initialize dataset
train_dataset = WavDataset(data=train_df, **dataset_params)
val_dataset = WavDataset(data=val_df, **dataset_params)
test_dataset = WavDataset(data=test_df, **dataset_params)

#set up DataLoader with custom collate
train_loader = DataLoader(dataset=train_dataset,batch_size=1,shuffle=True,collate_fn=collate_features, num_workers=0)
val_loader = DataLoader(dataset=val_dataset,batch_size=1,shuffle=False,collate_fn=collate_features, num_workers=0)
test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,collate_fn=collate_features, num_workers=s)

#initialize model
model = CustomAutoModel.from_pretrained(**model_params)

#training
trainer = Trainer(model=model, **training_params)
trainer.fit(train_loader, val_loader, epoch=1)
trainer.test(test_loader)
```

# Random Additional notes
- once we have models, determine clinical twist, try in just ALS?

a2-highgpu-2g
 

## TRAINING PARAMS
* FEATURES: hoarse_harsh, slow_rate, sound_distortions, monopitch_monoloudness, 'inappropriate_silences_or_prolonged_intervals'
* batch_size: 16
* learning_rate: 0.001, 0.01, 0.0001
* tf_learning_rate: 1e-6, 1e-5, 1e-4
* loss: rank
    * bce_weight: 0, 0.25, 0.5, 1
* number of classifier layers: TODO
* freezing/finetuning: (if you are going to add parameters to model - could add to classifier level, limited compute/limited data - where to add it? Add one more classifier layer and see what happens? More data you have, the more parameters you can reasonably optimize - we're always low data, so what's the best way to limit the data.)
    * all 
    * exclude-last 
    * half
    * required-only
    * LoRA
    * soft-prompting
    * add one layer to classifier
* pooling
    * mean
    * max
    * attention
* MODELS:
    * wavlm-large
    * hubert-large
    * whisper-medium
* scheduler: TODO
    * cosine, #skip warmup
    * do by epoch not training step bc it's a small thought

## QUESTIONS/RESEARCH
* CURRENT RUNNING TIME: LoRA - ~30min PER EPOCH
*  5 MOST COMMON FEATURES IN SENTENCE REPETITION - REDO WITH NEW DC EVENTUALLY TODO:
        * hoarse_harsh: 452
        * slow_rate: 402
        * sound_distortions: 349
        * monopitch_monoloudness: 341
        * inappropriate_silences_or_prolonged_intervals: 264
        
        ```
        data = pd.read_csv('CSV')
        data_feats = data[_FEATURES]
        freq = (data_feats > 1).sum()
        print(freq.sort_values()[-5:])
        ```


## All TODO
* visualizations 
    * training vs. validation loss
    * weights? 
    * any kind of attention heads during pooling?
    * outputs

