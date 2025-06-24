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
* what are our metrics of interest, just accuracy? balanced accuracy? anything else'
* LOSS FUNCTION - BCE 
* key and query in LoRA. common things - figure out what the standard things are for HuBERT/WavLM/Whisper. 
* don't pool over Padding tokens!!!!!!! in any pooling!!! how do we ensure that? what does that mean. 
* Leland - for one of the models that he trained, he came up with a combo loss that is BCE and w/in each batch, used more finegrained ratings to get the rank of the samples in that batch correctly (mildest to most severe, can learn mild to severe). Get splits that aren't yet binarized, add transform to binarize. 

## ACTIVE DEBUGGING/TASKS
* update splits
* normalize data?
* TEST SUITE - ALL ASSERTIONS!
    * ~~split~~
    * ~~hugging face models~~ (load smallest of each type and clear cache afterwards)
    * ~~classifier~~
    * ~~base model~~
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
    * loops/metrics

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
* random weight initialization for classifier - check seed
    * tests
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
    * decide basic configurations/layer options
    * decide test features vs. all features? different groupings? what is the plan here
* Add LoRA option
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
    * training + logging options
    * validation + logging options
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


