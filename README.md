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
* which features to use for debugging decide test features vs. all features? different groupings? what is the plan here
* don't pool over Padding tokens!!!!!!! in any pooling!!! how do we ensure that? what does that mean. 

## ACTIVE DEBUGGING/TASKS
* normalize data?
* add learning rate scheduler with warmup? more interesting learning rate options
* avg loss?
* metrics
* TEST SUITE
    * io/transforms - GCS test google cloud things????? mark to run only sometimes
    * loops/metrics - make sure logging works correctly
    * finetuning methods in trainer
    * test giving no rating threshold (assertion error)
    * test scheduler - determine good values for it
    * test early stop (add param for optional delta) (force early stopping?)
    * test giving non-existent loss (NotImplementedError)
    * test ranked classification loss
    * test new classifier params
    * try to load hubert checkpoint to wavlm model
* determine what classifier build to use with Rankings? It's not going to work the same as BCE loss...not entirely sure how to predict rank - multiclass/multicategory? ask leland what he did

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
* ~~Classification head~~
    * ~~decide basic configurations/layer options~~
    * ~~random weight initialization for classifier - check seed~~
    * ~~tests~~
* ~~Add LoRA option~~
    * ~~tests~~
* ~~Add soft prompting option~~
    * ~~test~~
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

