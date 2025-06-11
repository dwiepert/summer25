_MODELS = {  "wavlm-base": {
                  "hf_hub": "microsoft/wavlm-base",
                  "use_featext": True,
                  "in_features":768,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "trim": False
            },
            "wavlm-large": {
                  "hf_hub": "microsoft/wavlm-large",
                  "use_featext": True,
                  "in_features": 1024,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "trim": False
            },
            "hubert-base": {
                  "hf_hub": "facebook/hubert-base-ls960",
                  "use_featext": True,
                  "in_features":768,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "trim": False
            },
            "hubert-large": {
                  "hf_hub": "facebook/hubert-large-ll60k",
                  "use_featext": True,
                  "in_features":1024,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "trim": False
            },
            "hubert-xlarge": {
                  "hf_hub": "facebook/hubert-xlarge-ll60k",
                  "use_featext": True,
                  "in_features":1280,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "trim": False
            },
            "whisper-tiny": {
                  "hf_hub": "openai/whisper-tiny",
                  "use_featext": True,
                  "in_features":384,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'trim':True,
                  'clip_length': 30
            },
            "whisper-base": {
                  "hf_hub": "openai/whisper-base",
                  "use_featext": True,
                  "in_features":512,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'trim':True,
                  'clip_length': 30
            },
            "whisper-small": {
                  "hf_hub": "openai/whisper-small",
                  "use_featext": True,
                  "in_features":768,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'trim':True,
                  'clip_length': 30
            },
            "whisper-medium": {
                  "hf_hub": "openai/whisper-medium",
                  "use_featext": True,
                  "in_features":1024,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'trim':True,
                  'clip_length': 30
            },
            "whisper-large": {
                  "hf_hub": "openai/whisper-large",
                  "use_featext": True,
                  "in_features":1280,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'trim':True,
                  'clip_length': 30
            },
            "whisper-large-v2": {
                  "hf_hub": "openai/whisper-large-v2",
                  "use_featext": True,
                  "in_features":1280,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'trim':True,
                  'clip_length': 30
            }
}

_POOL = ['mean', 'max']
_FREEZE = ['all', 'layer', 'none']
_REQUIRED_ARGS =['model_type', 'pt_ckpt', 'seed', 'freeze_method', 'pool_method', 'out_features']