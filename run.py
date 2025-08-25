"""
Run experiments

Author(s): Daniela Wiepert
Last modified: 08/2025
"""
#IMPORTS
##built-in
import argparse
import json
from pathlib import Path
from typing import List

##third-party
import torch
from google.cloud import storage
from torch.utils.data import DataLoader

##local
from summer25.models import CustomAutoModel
from summer25.dataset import seeded_split, WavDataset, collate_features
from summer25.constants import _MODELS,_FREEZE, _FEATURES, _FINETUNE, _LOSS, _SCHEDULER, _OPTIMIZER
from summer25.training import Trainer
from summer25.io import search_gcs

_REQUIRED_MODEL_ARGS =['model_type']
_REQUIRED_LOAD = ['output_dir', 'audio_dir', 'split_dir']

# HELPER FUNCTIONS #
def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

# CHECK ARGUMENTS #
def check_load(args:dict) -> dict:
    """
    Load config file for data saving and audio loading, and ensure required arguments are there and pass assertions. 

    :param args: dictionary of arguments - vars(argparse object)
    :return args: updated args dictionary
    """ 
    if args['load_cfg']:
        load_cfg = args.pop('load_cfg')
        args.update(load_cfg)

    for r in _REQUIRED_LOAD:
        assert args[r], f'The required argument `{r}` was not given. Use `-h` or `--help` if information on the argument is needed.'
    
    assert args['audio_dir'], 'Audio directory not given.'
    assert args['split_dir'], 'Split dir not given.'
    assert args['output_dir'], 'Output directory not given.'

    if not isinstance(args['audio_dir'], Path): args['audio_dir'] = Path(args['audio_dir'])
    if not isinstance(args['split_dir'],Path): args['split_dir'] = Path(args['split_dir'])
    if not isinstance(args['output_dir'], Path): args['output_dir'] = Path(args['output_dir'])

    assert args['loss_type'] in _LOSS, 'Invalid loss type'
    if args['loss_type'] == 'rank':
        assert args['rating_threshold'], 'Must give rating threshold for rank loss'
    if args['scheduler_type']:
        assert args['scheduler_type'] in _SCHEDULER, 'Invalid scheduler type.'
        if args['scheduler_type'] == 'exponential':
            if not args['gamma']:
                assert args['end_lr'], 'Must give end_lr for exponential scheduler.'
                assert isinstance(args['end_lr'], float), 'Must give end_lr for exponential scheduler.'
        elif args['scheduler_type'] == 'warmup-cosine':
            assert args['warmup_epochs'], 'Must give warmup_epochs for warmup lr'
            assert isinstance(args['warmup_epochs'], int), 'Must give warmup_epochs for warmup lr'
    assert args['optim_type'] in _OPTIMIZER, 'Invalid optimizer type'

    if args['bucket_name']:
        assert args['project_name'], 'Must give project name'
        assert isinstance(args['project_name'], str),  'Must give project name'
        storage_client = storage.Client(project=args['project_name'])
        bucket = storage_client.bucket(args['bucket_name'])
        existing = search_gcs(args['audio_dir'], args['audio_dir'], bucket)
        assert existing != [], 'Given audio directory does not exist.'
    else:
        assert args['audio_dir'].exists(), 'Given audio directory does not exist.'
        args['split_dir'].mkdir(parents=True, exist_ok=True)
        args['output_dir'].mkdir(parents=True, exist_ok=True)
        bucket = None

    args['bucket'] = bucket

    return args

def check_model(args:dict) -> dict:
    """
    Check if a model config file exists or has been given, load in if it does, and ensure required arguments are there and pass assertions. 

    If the model config file is not an existing model config (as saved in the model class), make sure that it follows the format of those config files. See README for more information. TODO!

    :param args: dictionary of arguments - vars(argparse object)
    :return args: updated args dictionary
    """
    #check for mismatch between existing and not existing model cfg
    ec = [e for e in args['output_dir'].rglob('*model_config.json')] 
    if ec != [] and args['use_existing_cfg']:
        existing_cfg = load_config(ec[0])
    else:
        existing_cfg = None
        #raise NotImplementedError('Updating args with existing config is not implemented.')

    #TODO: compare existing model_cfg and given arguments? or just overwrite with existing arguments? 

    if args['model_cfg']:
        model_cfg = args.pop('model_cfg')
        args.update(model_cfg)

    if existing_cfg is not None:
        args.update(existing_cfg)
        
    for r in _REQUIRED_MODEL_ARGS:
        assert args[r], f'The required argument `{r}` was not given in a config file or in the command line. Use `-h` or `--help` if information on the argument is needed.'

    if args['clf_ckpt'] is not None:
        if not isinstance(args['clf_ckpt'], Path): args['clf_ckpt'] = Path(args['clf_ckpt'])
    if args['ft_ckpt'] is not None:
        if not isinstance(args['ft_ckpt'], Path): args['ft_ckpt'] = Path(args['ft_ckpt'])
    
    args['hf_model'] = False
    if 'hf_hub' in _MODELS[args['model_type']]: args['hf_model'] = True
        
    assert args['freeze_method'], 'Freeze method not given. Check command line arguments or model configuration file.'
    if args['freeze_method'] == 'layer': assert args['unfreeze_layers'], 'unfreeze_layers must be given if freeze method is `layer`.'
    
    assert args['finetune_method'], 'Finetune method not given. Check command line arguments or model configuration file.'
    
    if args['bucket']:
        assert args['pt_ckpt'], 'Must give pt_ckpt if loading from bucket'
    return args

def save_path(args:argparse.Namespace) -> argparse.Namespace:
    """
    Generate save path to ensure models are not overwritten 
    """
    print('Output directory save path not fully implemented.')
    return args

# PREP ARGUMENTS #
def zip_clf(args:argparse.Namespace) -> dict:
    """
    Zip arguments for classifier into a dictionary
    :param args: argparse Namespace object
    :return clf_args: dict of classifier arguments
    :return clf_ckpt: str, clf checkpoint
    """
    clf_args = {'separate':args.separate, 'clf_type':args.clf_type}
    if args.loss_type == 'rank':
        clf_args['binary'] = False
    else:
        clf_args['binary'] = True
    if args.target_features:
        clf_args['out_features'] = len(args.target_features)
    else:
        clf_args['out_features'] = len(_FEATURES)
    if args.nlayers:
        clf_args['nlayers'] = args.nlayers
    if args.activation:
        clf_args['activation'] = args.activation
    if args.clf_ckpt:
        clf_ckpt = args.clf_ckpt
    else:
        clf_ckpt = None
    if args.bottleneck:
        clf_args['bottleneck'] = args.bottleneck
    if args.layernorm:
        clf_args['layernorm'] = args.layernorm
    if args.dropout:
        clf_args['dropout'] = args.dropout
    if args.num_heads:
        clf_args['num_heads'] = args.num_heads
    
    return clf_args, clf_ckpt


def zip_model(args:argparse.Namespace) -> dict:
    """
    Zip arguments for base model into a dictionary
    :param args: argparse Namespace object
    :return model_args: dict of model arguments
    """
    to_add = f'e{args.epochs}_lr{args.learning_rate}_{args.optim_type}_{args.scheduler_type}'

    args.output_dir = args.output_dir / to_add
    if args.hf_model:
        model_args = {'model_type':args.model_type,'out_dir':args.output_dir,
                    'freeze_method':args.freeze_method, 'pool_method':args.pool_method,
                    'seed':args.seed,'finetune_method': args.finetune_method,  'normalize':args.normalize,
                    'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"), "print_memory":args.print_memory}
        d = model_args['device']
        print(f'Current device: {d}')
    else:
        raise NotImplementedError('Only compatible with huggingface models currently.')
    
    from_hub = args.from_hub
    if args.pt_ckpt:
        pt_ckpt = args.pt_ckpt
        from_hub = False
    else:
        pt_ckpt = None
    if args.ft_ckpt:
        ft_ckpt = args.ft_ckpt
    else:
        ft_ckpt = None
    if args.unfreeze_layers:
        model_args['unfreeze_layers'] = args.unfreeze_layers

    if args.lora_rank:
        model_args['lora_rank'] = args.lora_rank
    if args.lora_alpha:
        model_args['lora_alpha'] = args.lora_alpha
    if args.lora_dropout:
        model_args['lora_dropout'] = args.lora_dropout
    
    if args.virtual_tokens:
        model_args['virtual_tokens'] = args.virtual_tokens

    if args.bucket:
        model_args['bucket'] = args.bucket
        from_hub = False

    model_args['from_hub'] = from_hub
    out_args = {'config': model_args, 'ft_checkpoint':ft_ckpt, 'pt_checkpoint':pt_ckpt, 'delete_download':args.delete_download, 'data_parallel':args.data_parallel}
    return out_args

def zip_splits(args:argparse.Namespace) -> dict:
    """
    Zip arguments for splits into a dictionary
    :param args: argparse Namespace object
    :return split_args: dict of split arguments
    """
    split_args = {'audio_dir': args.audio_dir, 'split_dir': args.split_dir,
                  'seed': args.seed, 'save': args.save_split, 'load_existing': args.load_existing_split, 'as_json':args.as_json}

    assert args.subject_key
    assert args.date_key
    assert args.task_key
    assert args.audio_key
    split_args['subject_key'] = args.subject_key
    split_args['date_key'] = args.date_key
    split_args['task_key'] = args.task_key
    split_args['audio_key'] = args.audio_key
    if args.target_tasks:
        if isinstance(args.target_tasks, list):
            split_args['target_tasks'] = args.target_tasks
    if args.target_features:
        if isinstance(args.target_features, list):
            split_args['target_features'] = args.target_features
    if args.proportions:
        if isinstance(args.proportions, list):
            split_args['proportions'] = args.proportions
    if args.stratify_threshold:
        if isinstance(args.stratify_threshold, float):
            split_args['stratify_threshold'] = args.stratify_threshold
    if args.bucket:
        split_args['bucket'] = args.bucket
    return split_args

def zip_dataset(args:argparse.Namespace) -> dict:
    """
    Zip arguments for dataset
    :param args: argparse Namespace object
    :return dataset_args: dict of dataset arguments
    """
    dataset_args = {'prefix': args.audio_dir, 'model_type': args.model_type, 'uid_col':args.audio_key, 'target_labels': args.target_features,
                    'bucket': None, 'extension': args.audio_ext, 'structured': args.structured}
    
    if args.bucket:
        dataset_args['bucket'] = args.bucket
    if args.loss_type == 'rank':
        dataset_args['rank_prefix'] = args.rank_prefix

    if 'transforms' in args:
        transforms = args.transforms
    else:
        transforms = {}

    if args.use_librosa and 'use_librosa' not in transforms:
        transforms['use_librosa'] = args.use_librosa

    if 'truncate' not in transforms:
        if args.clip_length:
            truncate = {'length': args.clip_length}
            transforms['truncate'] = truncate
    
    if 'trim_level' not in transforms:
        if args.trim_level:
            transforms['trim_level'] = args.trim_level

    dataset_args['config'] = transforms
    return dataset_args

def zip_finetune(args):
    """
    Zip arguments for finetuning
    :param args: argparse Namespace object
    :return finetune_args: dict of finetune arguments
    """
    finetune_args = {'optim_type': args.optim_type, 'learning_rate':args.learning_rate, 'loss_type':args.loss_type,
                     'scheduler_type': args.scheduler_type, 'epochs': args.epochs, 'early_stop':args.early_stop}
    
    if args.tf_learning_rate:
        finetune_args['tf_learning_rate'] = args.tf_learning_rate

    if args.patience:
        finetune_args['patience'] = args.patience
    if args.delta:
        finetune_args['delta'] = args.delta

    if args.loss_type == 'rank':
        finetune_args['rating_threshold'] = args.rating_threshold
        finetune_args['margin'] = args.margin
        finetune_args['bce_weight'] = args.bce_weight
    
    if args.scheduler_type == 'exponential':
        if args.gamma:
            finetune_args['gamma'] = args.gamma
            if args.tf_gamma:
                finetune_args['tf_gamma'] = args.gamma
        else:
            finetune_args['end_lr'] = args.end_lr
            if args.end_tf_lr:
                finetune_args['end_tf_lr'] = args.end_tf_lr
            finetune_args['epochs'] = args.epochs
    elif args.scheduler_type == 'warmup-cosine':
        finetune_args['warmup_epochs'] = args.warmup_epochs


    if args.target_features:
        finetune_args['target_features'] = args.target_features
    else:
        finetune_args['target_features'] = _FEATURES

    if args.bucket:
        finetune_args['bucket'] = args.bucket

    return finetune_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Parser")
    #OPTIONAL CONFIG FILES
    cfg_args = parser.add_argument_group('cfg', 'configuration file related arguments')
    cfg_args.add_argument('--load_cfg', type=load_config, help='Audio loading configuration json')
    cfg_args.add_argument('--model_cfg', type=load_config, help="Model configuration json")
    cfg_args.add_argument('--use_existing_cfg', action='store_true', help='Specify whether to use an existing config file if it exists in the given output_dir')
    #I/O
    io_args = parser.add_argument_group('io', 'file related arguments')
    io_args.add_argument('--data_parallel', action='store_true', help='Specify whether using multiple gpus')
    io_args.add_argument('--bucket_name', type=str, help='Bucket name for GCS.')
    io_args.add_argument('--project_name', type=str, help='GCS project name.')
    io_args.add_argument('--audio_dir', type=Path, help='Directory with audio files & a csv with information on speakers/task.')
    io_args.add_argument('--audio_ext', type=str,default='wav', help='Audio extension.')
    io_args.add_argument('--structured', action='store_true', help='Specify whether the audio directory stores audio in structured manner (uid/waveform.wav) or not (uid.wav)')
    io_args.add_argument('--use_librosa', action='store_true', help='Specify whether to load audio with librosa')
    io_args.add_argument('--split_dir', type=Path, help='Directory with csv or jsons of file splits.')
    io_args.add_argument('--load_existing_split', action='store_true', help='Default to loading exisiting split.')
    io_args.add_argument('--as_json', action='store_true', help='True if loading/saving splits as json files.')
    io_args.add_argument('--save_split', action='store_true', help='Save generated data splits.')
    io_args.add_argument('--output_dir', type=Path, help='Output directory for saving all files.')
    io_args.add_argument('--target_tasks', nargs="+", type=List[str], default=['sentence_repetition'], help="Specify target tasks for dataset.")
    io_args.add_argument('--target_features', nargs="+", type=List[str], default=['hoarse_harsh', 'slow_rate', 'sound_distortions', 'monopitch_monoloudness', 'inappropriate_silences_or_prolonged_intervals'], help="Specify target features for dataset.")
    io_args.add_argument('--stratify_threshold', type=int, help="Specify minimum number of positive examples for a feature group.")
    io_args.add_argument('--subject_key', type=str, help='Specify column/key name for subjects in dataset metadata table')
    io_args.add_argument('--date_key', type=str, help='Specify column/key name for date in dataset metadata table')
    io_args.add_argument('--task_key', type=str, help='Specify column/key name for tasks in dataset metadata table')
    io_args.add_argument('--audio_key', type=str, help='Specify column/key name for audio file names in dataset metadata table')
    io_args.add_argument('--train_proportion', type=float, default=0.7, help='Specify split proportions.')
    io_args.add_argument('--val_proportion', type=float, default=0.15, help='Specify split proportions.')
    io_args.add_argument('--clip_length', type=float, help="Specify audio clip length in s.")
    io_args.add_argument('--trim_level', type=float, help="Specify silence trim level (dB for use_librosa, trigger level for torchaudio)")
    io_args.add_argument('--normalize', action='store_true', help='Specify whether to normalize audio.')
    #BASE MODEL
    model_args = parser.add_argument_group('model', 'model related arguments')
    model_args.add_argument('--model_type', type=str,
                            choices=list(_MODELS.keys()), help='Specify model type')
    model_args.add_argument('--pt_ckpt', type=Path, help='Specify local pretrained model path. Only required for hugging face models if issues loading from hub.')
    model_args.add_argument('--delete_download', action='store_true', help='Specify whether to delete local downloads.')
    model_args.add_argument('--from_hub', action='store_true', help='Specify whether to load from hub.')
    model_args.add_argument('--seed', type=int, help='Specify random seed for model initialization.')
    model_args.add_argument('--finetune_method', type=str, choices=_FINETUNE, help='Specify what finetuning method to use')
    model_args.add_argument('--lora_rank', type=int, help='If finetuning with lora, optionally give rank (default = 8)')
    model_args.add_argument('--lora_alpha', type=int, help='If finetuning with lora, optionally give alpha (default = 16)')
    model_args.add_argument('--lora_dropout', type=float, help='If finetuning with lora, optionally give dropout (default = 0.0)')
    model_args.add_argument('--virtual_tokens', type=int, help='If finetuning with soft prompting, optionally give number of tokens (default = 4)')
    model_args.add_argument('--freeze_method', type=str, choices=_FREEZE, help='Specify what freeze method to use.')
    model_args.add_argument('--unfreeze_layers', nargs="+", help="If freeze_method is `layer`, use this to specify which layers to freeze")
    model_args.add_argument('--ft_ckpt', type=Path, help='Specify finetuned model checkpoint')
    model_args.add_argument('--pool_method', type=str, choices=['mean', 'max', 'attn'], help='Specify pooling method prior to classification head.')
    #CLASSIFIER
    clf_args = parser.add_argument_group('classifier', 'classifier related arguments')
    clf_args.add_argument('--clf_type', type=str, default='linear', help='Specify whether to use linear classifier or transformer classifier.')
    clf_args.add_argument('--nlayers', type=int, help='Specify classification head size.')
    clf_args.add_argument('--activation', type=str, help='Specify type of activation function to use in the classifier.')
    clf_args.add_argument('--bottleneck', type=int, help='Specify optional bottleneck if nlayers >= 2.')
    clf_args.add_argument('--dropout', type=float, help='Specify optional dropout rate.')
    clf_args.add_argument('--layernorm', action='store_true', help='Specify whether to add layernorm to classifier.')
    clf_args.add_argument('--clf_ckpt', type=Path, help="Specify classification checkpoint.")
    clf_args.add_argument('--separate', action='store_true', help='Specify whether to use separate classifiers.' )
    clf_args.add_argument('--num_heads', type=int, help='Specify number of heads in transformer classifier.')
    #TRAINING ARGS
    train_args = parser.add_argument_group('train', 'training/testing related arguments')
    train_args.add_argument('--batch_sz', default=1, type=int, help='Set batch size.')
    train_args.add_argument('--num_workers', default=0, type=int, help='Set num workers for DataLoader.')
    train_args.add_argument('--optim_type', type=str, default='adamw', choices=_OPTIMIZER, help='Specify type of optimizer to use. (default = adamw)')
    train_args.add_argument('--learning_rate', type=float, default=1e-3, help='Specify learning rate (default=1e-3)')
    train_args.add_argument('--tf_learning_rate', type=float, help='Optionally specify transformer specific learning rate')
    train_args.add_argument('--loss_type', type=str, default='bce', choices=_LOSS, help='Specify type of optimizer to use. (default = bce)')
    train_args.add_argument('--rank_prefix', type=str, default='rank_', help='Specify prefix for columns with rank targets if using rank loss.')
    train_args.add_argument('--rating_threshold', type=float, help='Specify rating threshold for rank loss.')
    train_args.add_argument('--margin', type=float,default=1.0, help='Specify margin for rank loss.')
    train_args.add_argument('--bce_weight', type=float,default=0.5, help='Specify weighting of BCE for rank loss.')
    train_args.add_argument('--scheduler_type', type=str, choices=_SCHEDULER, help='Specify type of scheduler to use.')
    train_args.add_argument('--end_lr', type=float, help='Specify end learning rate if using exponential scheduler')
    train_args.add_argument('--end_tf_lr', type=float, help='Optionally specify end learning rate for transformer if using exponential scheduler')
    train_args.add_argument('--gamma', type=float, help='Specify gamma if using exponential scheduler')
    train_args.add_argument('--tf_gamma', type=float, help='Optionally specify gamma for transformer if using exponential scheduler')
    train_args.add_argument('--warmup_epochs', type=int, help='Specify warmup_epochs if using warmup scheduler.')
    train_args.add_argument('--tf_warmup_epochs', type=int, help='Optionally specify warmup_epochs for transformer if using warmup scheduler.')
    train_args.add_argument('--early_stop', action='store_true', help='Specify whether to use early stopping.')
    train_args.add_argument('--patience', type=int, help='Specify patience for early stopping. (default = 5)')
    train_args.add_argument('--delta', type=float, help='Specify delta for early stopping.')
    train_args.add_argument('--epochs', type=int, default=1, help='Specify epochs for finetuning. (default = 1)')
    train_args.add_argument('--eval_only', action='store_true', help='Specify whether to only run evaluation.')
    train_args.add_argument('--print_memory', action='store_true', help='Specify whether to print memory.' )
    train_args.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    args.proportions = [args.train_proportion, args.val_proportion, (1-args.train_proportion-args.val_proportion)]

    args_dict = vars(args)
    args_dict_l = check_load(args_dict)
    args_dict_m = check_model(args_dict_l)
    updated_args = save_path(argparse.Namespace(**args_dict_m))
    
    ## INITIALIZE EXTRACTOR AND MODEL
    ma = zip_model(updated_args)
    ca, cckpt = zip_clf(updated_args)
    temp_config = ma['config']
    temp_config.update(ca)
    ma['config'] = temp_config
    ma['clf_checkpoint'] = cckpt

    if updated_args.bucket:
        assert 'bucket' in ma['config']
    else:
        print('No bucket available')
        
    #INITIALIZE MODEL
    model, feature_extractor = CustomAutoModel.from_pretrained(**ma)
    

    ## DATA
    sa = zip_splits(updated_args)
    da = zip_dataset(updated_args)

    if updated_args.bucket:
        assert 'bucket' in sa
        assert 'bucket' in da
    else:
        print('No bucket available')

    #DATA SPLIT
    train_df, val_df, test_df = seeded_split(**sa)

    if args.debug:
        train_df = train_df[:10]
        val_df = val_df[:10]
        test_df = test_df[:10]
    
    train_dataset = WavDataset(data=train_df, feature_extractor=feature_extractor, **da)
    test_dataset = WavDataset(data=test_df, feature_extractor=feature_extractor, **da)
    val_dataset = WavDataset(data=val_df,feature_extractor=feature_extractor, **da)

    #using custom collate
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_sz,shuffle=True,collate_fn=collate_features, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset,batch_size=args.batch_sz,shuffle=False,collate_fn=collate_features, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset,batch_size=args.batch_sz,shuffle=False,collate_fn=collate_features, num_workers=args.num_workers)
    
    ## FINETUNING
    fa = zip_finetune(updated_args) #don't forget extra scheduler args

    if updated_args.bucket:
        assert 'bucket' in fa
    else:
        print('No bucket available')
    
    if fa['scheduler_type']:
        if 'cosine' in fa['scheduler_type']:
            fa['train_len'] = len(train_df) # for cosine annealing scheduler
    ## TRAIN MODEL
    model_trainer = Trainer(model=model, **fa)
    if not args.eval_only:
        model_trainer.fit(train_loader, val_loader, epochs=args.epochs)
        
    model_trainer.test(test_loader)
   

