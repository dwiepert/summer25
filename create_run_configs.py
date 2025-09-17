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
from pathlib import Path
import json

parser = argparse.ArgumentParser(description="Main Parser")
parser.add_argument('--base_cfg', type=str)
parser.add_argument('--cfg_path', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--audio_dir', type=str)
parser.add_argument('--split_dir', type=str)
parser.add_argument('--bucket_name', type=str)
parser.add_argument('--project_name', type=str)
parser.add_argument('--pt_checkpoint_root', type=str)
parser.add_argument('--subject_key', type=str)
parser.add_argument('--date_key', type=str)
parser.add_argument('--audio_key', type=str)
parser.add_argument('--task_key', type=str)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument
args = parser.parse_args()

#create an args list
args.cfg_path = Path(args.cfg_path)
with open(args.base_cfg, 'r') as file:
    base_cfg = json.load(file)

def set_args(cfg, args, out_path):
    worker_pool_specs = cfg['workerPoolSpecs'][0]
    container = worker_pool_specs['containerSpec']
    container["args"] = args
    worker_pool_specs['containerSpec'] = container
    cfg['workerPoolSpecs'] = [worker_pool_specs]

    with open(out_path, "w") as file:
        json.dump(cfg, file, indent=4)  # 'indent' makes the file more 

base_arg_list = [   f"--output_dir={args.output_dir}",
                    f"--audio_dir={args.audio_dir}",
                    f"--split_dir={args.split_dir}",
                    f"--bucket_name={args.bucket_name}",
                    f"--project_name={args.project_name}",
                    "--load_existing_split",
                    f"--subject_key={args.subject_key}",
                    f"--date_key={args.date_key}",
                    f"--audio_key={args.audio_key}",
                    f"--task_key={args.task_key}",
                    "--stratify_threshold=10",
                    "--batch_sz=2",
                    "--loss_type=rank",
                    "--rating_threshold=2.0",
                    "--early_stop",
                    "--scheduler_type=cosine",
                    f"--epochs={args.epochs}",
                    "--clf_type=linear",
                    "--separate",
                    "--activation=relu",
                    "--gradient_accumulation_steps=4",
                    "--clip_length=15",
                    f"--patience={args.epochs}",
                    "--target_tasks",
                    "reading"
            ]


learning_rates = [1e-4] #--learning_rate
tf_learning_rates = [1e-6] #--tf_learning_rate
model_type = ['wavlm-large', 'hubert-large', 'whisper-medium'] #--model_type
args.pt_checkpoint_root = Path(args.pt_checkpoint_root)
checkpoints = [str(args.pt_checkpoint_root / m) for m in model_type] #--pt_ckpt
nlayers = [2]#,1,3] #--nlayers
freeze = ['all', 'exclude-last'] #, 'half'] #--freeze_method
finetune = ['lora', 'soft-prompt'] #--finetune_method
pool = ['mean', 'attn'] #--pool_method max
bce_weight = [0.5, 1] #0.25
seed = [100]#, 42]#, 56]



#### CLASSIFIER ONLY - multiple layers, freeze vs. unfreeze
sub_dir = 'clf_tests'
for i in range(len(model_type)):
    mt = model_type[i]
    ckpt = checkpoints[i]
    tflr = tf_learning_rates[0]
    lr = learning_rates[0]
    for nl in nlayers: # for each layer
        for f in freeze:
            for p in pool:
                for b in bce_weight:
                    for s in seed:
                        test_name = f'seed{s}_tflr{tflr}_lr{lr}_{mt}_nlayers{nl}_freeze{f}_pool{p}_bw{b}'
                        args_list = base_arg_list.copy()
                        args_list.append(f"--learning_rate={lr}")
                        args_list.append(f"--tf_learning_rate={tflr}")
                        args_list.append(f"--model_type={mt}")
                        args_list.append(f"--pt_ckpt={ckpt}")
                        args_list.append(f"--nlayers={nl}")
                        args_list.append(f"--freeze_method={f}") 
                        args_list.append(f"--finetune_method=none")
                        args_list.append(f"--pool_method={p}")
                        args_list.append(f"--bce_weight={b}")
                        args_list.append(f"--seed={s}")
                        out_path = args.cfg_path / sub_dir
                        out_path.mkdir(parents=True, exist_ok=True)
                        out_path = str(out_path / f"{test_name}.json")
                        set_args(base_cfg, args_list, out_path)

# #### finetune - multiple layers, freeze vs. unfreeze
# sub_dir = 'ft_tests_v2'
# for i in range(len(model_type)):
#     if i == 0:
#         mt = model_type[i]
#         ckpt = checkpoints[i]
#         tflr = tf_learning_rates[0]
#         lr = learning_rates[0]
#         for nl in nlayers: # for each layer
#             for f in finetune:
#                 for p in pool:
#                     for b in bce_weight:
#                         test_name = f'seed{seed[0]}_tflr{tflr}_lr{lr}_{mt}_nlayers{nl}_finetune{f}_pool{p}_bw{b}'
#                         args_list = base_arg_list.copy()
#                         args_list.append(f"--learning_rate={lr}")
#                         args_list.append(f"--tf_learning_rate={tflr}")
#                         args_list.append(f"--model_type={mt}")
#                         args_list.append(f"--pt_ckpt={ckpt}")
#                         args_list.append(f"--nlayers={nl}")
#                         args_list.append(f"--freeze_method=all") 
#                         args_list.append(f"--finetune_method={f}")
#                         args_list.append(f"--pool_method={pool[0]}")
#                         args_list.append(f"--bce_weight={bce_weight[0]}")
#                         args_list.append(f"--seed={seed[0]}")
#                         out_path = args.cfg_path / sub_dir
#                         out_path.mkdir(parents=True, exist_ok=True)
#                         out_path = str(out_path / f"{test_name}.json")
#                         set_args(base_cfg, args_list, out_path)

# sub_dir = 'lr_test_v2'
# for i in range(len(model_type)): #run for each model type
#     if i == 0:
#         mt = model_type[i]
#         ckpt = checkpoints[i] 
#         for tflr in tf_learning_rates:
#             for lr in learning_rates: #run for each learning rate 
#                 test_name = f'seed{seed[0]}_tflr{tflr}_lr{lr}_{mt}' #_nlayers{l}_freeze{f}_pool{p}_bw{b}'
#                 args_list = base_arg_list.copy()
#                 args_list.append(f"--learning_rate={lr}")
#                 args_list.append(f"--tf_learning_rate={tflr}")
#                 args_list.append(f"--model_type={mt}")
#                 args_list.append(f"--pt_ckpt={ckpt}")
#                 args_list.append(f"--nlayers={3}")
#                 args_list.append(f"--freeze_method=exclude-last") 
#                 args_list.append(f"--finetune_method=none")
#                 args_list.append(f"--pool_method={pool[0]}")
#                 args_list.append(f"--bce_weight={bce_weight[0]}")
#                 args_list.append(f"--seed={seed[0]}")

#                 out_path = args.cfg_path / sub_dir
#                 out_path.mkdir(parents=True, exist_ok=True)
#                 out_path = str(out_path / f"{test_name}.json")
#                 set_args(base_cfg, args_list, out_path)


# sub_dir = 'freeze_test'
# for s in seed: # each of these needs to be run for each seed 
#     for i in range(len(model_type)): #run for each model type
#         mt = model_type[i]
#         ckpt = checkpoints[i] 
#         for lr in learning_rates: #run for each learning rate 
#             for l in nlayers: #run for different numbers of clf layers 
#                 for f in freeze:
#                     ###### if unfreezing: for f in freeze:
#                     ##### if finetuning: for f in finetune: 
#                     for p in pool: # for different pooling types
#                         for b in bce_weight: # for different bce weights
#                             test_name = f'seed{s}_lr{lr}_{mt}_nlayers{l}_freeze{f}_pool{p}_bw{b}'
#                             args_list = base_arg_list.copy()
#                             args_list.append(f"--learning_rate={lr}")
#                             args_list.append(f"--tf_learning_rate={tf_learning_rates[0]}")
#                             args_list.append(f"--model_type={mt}")
#                             args_list.append(f"--pt_ckpt={ckpt}")
#                             args_list.append(f"--nlayers={l}")
#                             args_list.append(f"--freeze_method={f}") 
#                             args_list.append(f"--finetune_method=none")
#                             args_list.append(f"--pool_method={p}")
#                             args_list.append(f"--bce_weight={b}")
#                             args_list.append(f"--seed={s}")

#                             out_path = args.cfg_path / sub_dir
#                             out_path.mkdir(parents=True, exist_ok=True)
#                             out_path = str(out_path / f"{test_name}.json")
#                             set_args(base_cfg, args_list, out_path)

# sub_dir = 'finetune_test'
# for s in seed: # each of these needs to be run for each seed 
#     for i in range(len(model_type)): #run for each model type
#         mt = model_type[i]
#         ckpt = checkpoints[i] 
#         for lr in learning_rates: #run for each learning rate 
#             for l in nlayers: #run for different numbers of clf layers 
#                 ###### if unfreezing: for f in freeze:
#                 ##### if finetuning: for f in finetune: 
#                 for f in finetune:
#                     if f != 'none':
#                         for p in pool: # for different pooling types
#                             for b in bce_weight: # for different bce weights
#                                 test_name = f'seed{s}_lr{lr}_{mt}_nlayers{l}_ft{f}_pool{p}_bw{b}'
#                                 args_list = base_arg_list.copy()
#                                 args_list.append(f"--learning_rate={lr}")
#                                 args_list.append(f"--tf_learning_rate={tf_learning_rates[0]}")
#                                 args_list.append(f"--model_type={mt}")
#                                 args_list.append(f"--pt_ckpt={ckpt}")
#                                 args_list.append(f"--nlayers={l}")
#                                 args_list.append(f"--freeze_method=all") 
#                                 args_list.append(f"--finetune_method={f}")
#                                 args_list.append(f"--pool_method={p}")
#                                 args_list.append(f"--bce_weight={b}")
#                                 args_list.append(f"--seed={s}")

#                                 out_path = args.cfg_path / sub_dir
#                                 out_path.mkdir(parents=True, exist_ok=True)
#                                 out_path = str(out_path / f"{test_name}.json")
#                                 set_args(base_cfg, args_list, out_path)

""" # TEST - unfreezing


# TEST - lora 


# TEST - soft prompt


# TEST different learning rates for 1 model, keeping everything else the same
sub_dir = 'learning_rates'
test_name = 'lr'
for lr in learning_rates:
    for tflr in tf_learning_rates:
        test_name = f'lr{lr}_tflr{tflr}'
        args_list = base_arg_list.copy()
        args_list.append(f"--learning_rate={lr}")
        args_list.append(f"--tf_learning_rate={tflr}")
        args_list.append(f"--model_type={model_type[0]}")
        args_list.append(f"--pt_ckpt={checkpoints[0]}")
        args_list.append(f"--nlayers={nlayers[0]}")
        args_list.append(f"--freeze_method={freeze[0]}")
        args_list.append(f"--finetune_method={finetune[0]}")
        args_list.append(f"--pool_method={pool[0]}")
        args_list.append(f"--bce_weight={bce_weight[0]}")
        args_list.append(f"--seed={seed[0]}")

        out_path = args.cfg_path / sub_dir
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = str(out_path / f"{test_name}.json")
        set_args(base_cfg, args_list, out_path)

 """

        