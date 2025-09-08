"""
Various constants for ensuring consistency in model loading/training

Author(s): Daniela Wiepert
Last modified: 07/2025
"""
#IMPORTS
##third-party
from transformers import WavLMModel, WhisperModel, HubertModel

_MODELS = {  "wavlm-base": {
                  "model_instance": WavLMModel,
                  "hf_hub": "microsoft/wavlm-base",
                  "use_featext": True,
                  "in_features":768,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "monochannel": True,
                  "required_freeze": ['masked_spec_embed', 'feature_extractor', 'feature_projector'],
                  "optional_freeze": ['encoder.pos_conv_embed', 'encoder.layer_norm'],
                  "unfreeze_prefixes": ['encoder.layers'],
                  "lora_layers": ['k_proj','v_proj','q_proj'],
                  "attention_mask": False
            },
            "wavlm-large": {
                  "model_instance": WavLMModel,
                  "hf_hub": "microsoft/wavlm-large",
                  "use_featext": True,
                  "in_features": 1024,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "monochannel": True,
                  "required_freeze": ['masked_spec_embed', 'feature_extractor', 'feature_projector'],
                  "optional_freeze": ['encoder.pos_conv_embed', 'encoder.layer_norm'],
                  "unfreeze_prefixes": ['encoder.layers'],
                  "lora_layers": ['k_proj','v_proj','q_proj'],
                  "attention_mask": False
            },
            "hubert-base": {
                  "model_instance": HubertModel,
                  "hf_hub": "facebook/hubert-base-ls960",
                  "use_featext": True,
                  "in_features":768,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "monochannel": True,
                  "required_freeze": ['masked_spec_embed', 'feature_extractor', 'feature_projector'],
                  "optional_freeze": ['encoder.pos_conv_embed', 'encoder.layer_norm'],
                  "unfreeze_prefixes": ['encoder.layers'],
                  "lora_layers": ['k_proj','v_proj','q_proj'],
                  "attention_mask": False
            },
            "hubert-large": {
                  "model_instance": HubertModel,
                  "hf_hub": "facebook/hubert-large-ll60k",
                  "use_featext": True,
                  "in_features":1024,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "monochannel": True,
                  "required_freeze": ['masked_spec_embed', 'feature_extractor', 'feature_projector'],
                  "optional_freeze": ['encoder.pos_conv_embed', 'encoder.layer_norm'],
                  "unfreeze_prefixes": ['encoder.layers'],
                  "lora_layers": ['k_proj','v_proj','q_proj'],
                  "attention_mask": False
            },
            "hubert-xlarge": {
                  "model_instance": HubertModel,
                  "hf_hub": "facebook/hubert-xlarge-ll60k",
                  "use_featext": True,
                  "in_features":1280,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "monochannel": True,
                  "required_freeze": ['masked_spec_embed', 'feature_extractor', 'feature_projector'],
                  "optional_freeze": ['encoder.pos_conv_embed', 'encoder.layer_norm'],
                  "unfreeze_prefixes": ['encoder.layers'],
                  "lora_layers": ['k_proj','v_proj','q_proj'],
                  "attention_mask": False
            },
            "whisper-tiny": {
                  "model_instance": WhisperModel,
                  "hf_hub": "openai/whisper-tiny",
                  "use_featext": True,
                  "in_features":384,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "monochannel": True,
                  "required_freeze": ['encoder.conv1', 'encoder.conv2', 'encoder.embed_positions', 'decoder'],
                  "optional_freeze": [],
                  "unfreeze_prefixes": ['encoder.layers', 'encoder.layer_norm'],
                  "lora_layers": ['k_proj','v_proj','q_proj'],
                  "attention_mask": False
            },
            "whisper-base": {
                  "model_instance": WhisperModel,
                  "hf_hub": "openai/whisper-base",
                  "use_featext": True,
                  "in_features":512,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "monochannel": True,
                  "required_freeze": ['encoder.conv1', 'encoder.conv2', 'encoder.embed_positions', 'decoder'],
                  "optional_freeze": [],
                  "unfreeze_prefixes": ['encoder.layers', 'encoder.layer_norm'],
                  "lora_layers": ['k_proj','v_proj','q_proj'],
                  "attention_mask": False
            },
            "whisper-small": {
                  "model_instance": WhisperModel,
                  "hf_hub": "openai/whisper-small",
                  "use_featext": True,
                  "in_features":768,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "monochannel": True,
                  "required_freeze": ['encoder.conv1', 'encoder.conv2', 'encoder.embed_positions', 'decoder'],
                  "optional_freeze": [],
                  "unfreeze_prefixes": ['encoder.layers', 'encoder.layer_norm'],
                  "lora_layers": ['k_proj','v_proj','q_proj'],
                  "attention_mask": False
            },
            "whisper-medium": {
                  "model_instance": WhisperModel,
                  "hf_hub": "openai/whisper-medium",
                  "use_featext": True,
                  "in_features":1024,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "monochannel": True,
                  "required_freeze": ['encoder.conv1', 'encoder.conv2', 'encoder.embed_positions', 'decoder'],
                  "optional_freeze": [],
                  "unfreeze_prefixes": ['encoder.layers', 'encoder.layer_norm'],
                  "lora_layers": ['k_proj','v_proj','q_proj'],
                  "attention_mask": False
            },
            "whisper-large": {
                  "model_instance": WhisperModel,
                  "hf_hub": "openai/whisper-large",
                  "use_featext": True,
                  "in_features":1280,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "monochannel": True,
                  "required_freeze": ['encoder.conv1', 'encoder.conv2', 'encoder.embed_positions', 'decoder'],
                  "optional_freeze": [],
                  "unfreeze_prefixes": ['encoder.layers', 'encoder.layer_norm'],
                  "lora_layers": ['k_proj','v_proj','q_proj'],
                  "attention_mask": False
            },
            "whisper-large-v2": {
                  "model_instance": WhisperModel,
                  "hf_hub": "openai/whisper-large-v2",
                  "use_featext": True,
                  "in_features":1280,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "monochannel": True,
                  "required_freeze": ['encoder.conv1', 'encoder.conv2', 'encoder.embed_positions', 'decoder'],
                  "optional_freeze": [],
                  "unfreeze_prefixes": ['encoder.layers', 'encoder.layer_norm'],
                  "lora_layers": ['k_proj','v_proj','q_proj'],
                  "attention_mask": False
            },
            "test_model": {
                  "use_featext": False,
                  "in_features":1280,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "required_freeze": [],
                  "optional_freeze": [],
                  "unfreeze_prefixes": [],
                  "attention_mask": False

            },
            "test_model2": {
                  "use_featext": False,
                  "in_features":1280,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "required_freeze": [],
                  "optional_freeze": [],
                  "unfreeze_prefixes": [],
                  "attention_mask": False
            }
}

_POOL = ['mean', 'max', 'attention']
_FREEZE = ['all', 'half', 'exclude-last', 'layer', 'optional', 'required-only']
_FINETUNE = ['lora', 'soft-prompt', 'none']
_TASKS = ['sentence_repetition', 'word_repetition', 'reading', 'picture_description']
_LOSS = ['bce', 'rank']
_SCHEDULER = ['exponential', 'warmup-cosine', 'cosine']
_OPTIMIZER = ['adamw']
_FEATURES = ['abn_pitch_or_loudness_variability', 'abn_resonance',
       'accelerating_rate', 'audible_false_starts_or_restarts',
       'audible_nasal_emission', 'breathy', 'excess_equal_stress',
       'high_pitch', 'hoarse_harsh', 'hypernasal',
       'inappropriate_silences_or_prolonged_intervals', 'intelligibility',
       'irregular_artic_breakdowns', 'language', 'loudness_decay', 'low_pitch',
       'monopitch_monoloudness', 'other_cognitive_communication_skills',
       'prolonged_sounds', 'rapid_rate', 'reduced_syllables_per_breath_group',
       'repeated_sounds', 'slow_rate', 'sound_additions', 'sound_distortions',
       'sound_omissions', 'sound_sequencing_errors', 'sound_substitutions',
       'strained', 'stridor', 'syllable_segmentation',
       'telescoping_of_syllables', 'verbal_asides', 'vocal_noises',
       'voice_interruptions', 'word_or_phrase_repetitions']

