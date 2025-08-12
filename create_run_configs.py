"""

* freezing/finetuning: (if you are going to add parameters to model - could add to classifier level, limited compute/limited data - where to add it? Add one more classifier layer and see what happens? More data you have, the more parameters you can reasonably optimize - we're always low data, so what's the best way to limit the data.)
    * all 
    * exclude-last 
    * half
    * required-only
    * LoRA
    * soft-prompting
    * add one layer to classifier

"""
import argparse


parser = argparse.ArgumentParser(description="Main Parser")
parser.add_argument('--output_dir', type=str)
parser.add_argument('--audio_dir', type=str)
parser.add_argument('--split_dir', type=str)
args = parser.parse_args()


#load_cfg
load_cfg = {"output_dir": args.output_dir,
            "audio_dir": args.audio_dir,
            "split_dir": args.split_dir,
            "load_existing_split": True,
            "save_split": True,
            "as_json": False,
            "target_tasks": ["sentence_repetition"],
            "target_features": ["hoarse_harsh", "slow_rate", "sound_distortions", "monopitch_monoloudness", "inappropriate_silences_or_prolonged_intervals"],
            "subject_key": "subject",
            "date_key": "incident_date",
            "audio_key": "original_audio_id",
            "task_key": "task_name",
            "stratify_threshold": 10,
            "proportions": [0.8,0.1,0.1],
            "transforms": {},
            "batch_sz": 16,
            "loss_type": "rank",
            "rating_threshold": 2.0,
            "early_stop": True,
            "scheduler_type":"cosine",
            "epochs": 10,
            "debug":False}

#Varying versions
epochs = [10]
bce_weight = [0, 0.25, 0.5, 1]
tf_learning_weight = [1e-6, 1e-5, 1e-4]
pool_method = ['mean', 'max', 'attn']
model_type = ['wavlm-large', 'hubert-large', 'whisper-medium']
nlayers = [1, 2, 3] 
learning_rate = [0.001, 0.01, 0.0001]
freeze_method = ['all', 'exclude-last', 'half']
finetune_method = ['none', 'lora', 'soft-prompt']
