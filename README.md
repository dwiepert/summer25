# Summer 2025 Project

# Step 1:
Load hugging face models in flexible classes, add functionality to freeze as desired 

TODO: look at previous work/lastwish/etc. to see what you want to do for dataset/dataloader and audio loading

FREEZING FEATURE EXTRACTOR?

TO DEBUG:
* diff model types, diff freeze methods, diff pool methods
* unfreeze_layers, freeze_extractor, freezing in general 
* pool_dim
* clf_ckpt, ft_ckpt, pt_ckpt
* download w hugging face hub local, pt_ckpt
* load existing configuration

CURRENT DEBUG TODOS:
* keep_extractor for models without feature extractor!
PLAN:
* load from existing configuration
* ~~load huggingface models~~ 
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper (make sure it recognizes whisper separately)~~
    * ~~Check hugging face hub download and load - + check delete_post (change to delete_download)~~
    * ~~Check local checkpoint load~~
    * ~~confirm configuration is saving properly~~
* random weight initialization - check seed for classifier and base model
* processor - tokenizer needed?
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper~~
* freeze by layers (decide layer configurations) for -  assert int or str
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper~~ - should whisper decoder always be frozen since we aren't using it at all?^^
    * keep extractor
        * hubert???
        * wavlm
* Classification head 
    * decide basic configurations/layer options
    * decide test features vs. all features? different groupings? what is the plan here
* Add LoRA option
* Add soft prompting option 
* Flexibly create data splits - each seed has a different split? 
    * split by randomly selecting 
    * split by speaker
    * split by task
* Load data into a dataset
    * transform to 16000 if not done
    * optional trim (FOR WHISPER!!)
    * convert to tensor
    * normalize data?
    * check batching collate_fn
    * use model specific processors????
* pooling strategies
    * determine what pool dimension you need
    * mean
    * max
    * attention
* test run forward pass of models
    * do we need model-specific feature processors????
    * wavlm
    * hubert
    * whisper^^
* CONFIRM METRICS
* model loops
    * training + logging options - WHAT IS THE BEST LOSS? OPTIMIZER? ETC.
    * validation + logging options
    * save best model (see other code for tracking the best model) + final model? 
    * testing - what testing do we want to do 
* check finetuned checkpoint loading
    * base model
    * classification head
* visualizations 
    * training vs. validation loss
    * weights? 
    * any kind of attention heads during pooling?
    * outputs


