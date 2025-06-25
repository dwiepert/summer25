# Summer 2025 Project
REQUIREMENTS:
    audio directory format
    metadata expectations

CONFIG FILES:
    - run.py only contains some parameters for audio
    - can specify more exact specifics for WavDataset transforms by adding to a config file. TODO

## QUESTIONS
* Extra layer norm in whisper????
* We don't actually want to re-initialize any weights except to randomly initialize classifier weights, correct? 
* safe for features to be public? tasks to be public?
* which features to use for debugging
* don't pool over Padding tokens!!!!!!! in any pooling!!! how do we ensure that? what does that mean. 
* i'm not sure soft prompting is what we want actually, and i'm not sure how to add it effectively

## ACTIVE DEBUGGING/TASKS
* normalize data?
* TEST SUITE - ALL ASSERTIONS!
    * ~~split~~
    * ~~hugging face models~~ TEST FINETUNE METHOD
    * ~~classifier~~ TEST RANDOM WEIGHTS + NEWLY ADDED ARGUMENTS
    * ~~base model~~ TEST FINETUNE METHOD
    * ~~base dataset~~
    * ~~io/transforms - Non-GCS~~
    * io/transforms - GCS test google cloud things????? mark to run only sometimes
    * ~~wav dataset~~
    * ~~HF extractor~~
    * ~~freeze~~ 
    * ~~pad transform~~
    * ~~pool~~
    * ~~custom collate~~
    * ~~forward~~
    * loops/metrics - make sure logging works correctly
    * lora
    * new model saving method (save_pretrain)
    * new model loading method
    * add learning rate scheduler with warmup?

## All TODO
* Make seeded_split compatible with gcs + run
* ~~load from existing configuration~~
* ~~load huggingface models~~ 
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper (make sure it recognizes whisper separately)~~
    * ~~Check hugging face hub download and load - + check delete_post (change to delete_download)~~
    * ~~Check local checkpoint load~~
    * ~~confirm configuration is saving properly~~
    * ~~tests~~
* ~~processor - tokenizer needed?~~
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper~~
* ~~freeze by layers (decide layer configurations) for -  assert int or str~~
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper~~
    * ~~set module names - can then just give layers? determine which modules to freeze~~
    * ~~tests~~
* Classification head 
    * ~~decide basic configurations/layer options~~
    * decide test features vs. all features? different groupings? what is the plan here
    * random weight initialization for classifier - check seed
    * tests
* Add LoRA option
    * tests
* Add soft prompting option 
* ~~Flexibly create data splits - each seed has a different split?~~ 
* ~~Load data into a dataset~~
    * ~~transform to 16000 if not done~~
    * ~~optional trim (FOR WHISPER!!)~~
    * ~~convert to tensor~~
    * ~~normalize data?^^~~
    * ~~check batching collate_fn~~
    * ~~use model specific processors????~~
* ~~pooling strategies~~
    * ~~mean~~
    * ~~max~~
    * ~~attention~~
    * ~~tests~~
* ~~test run forward pass of models~~
    * ~~do we need modsel-specific feature processors????~~
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper~~
    * ~~tests (include classifier only)~~
* CONFIRM METRICS
* model loops
    * ~~training + logging options~~
    * ~~validation + logging options~~
    * ~~save best model (see other code for tracking the best model) + final model?~~ 
    * testing - what testing do we want to do 
* ~~check finetuned checkpoint loading~~
    * ~~base model~~
    * ~~classification head~~
* visualizations 
    * training vs. validation loss
    * weights? 
    * any kind of attention heads during pooling?
    * outputs


