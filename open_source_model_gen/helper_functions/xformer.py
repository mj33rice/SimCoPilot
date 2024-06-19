import logging

import numpy as np
import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import RobertaTokenizer, T5Tokenizer, BartTokenizer, GPT2Tokenizer, OpenAIGPTTokenizer, \
    BertTokenizer, DistilBertTokenizer, AutoTokenizer


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    # ############################# Microsoft Phi Models ############################## #
    'phi-3-mini-4k': (AutoConfig, AutoModelForCausalLM),
    'phi-3-small-8k': (AutoConfig, AutoModelForCausalLM),
    'phi-3-medium-4k': (AutoConfig, AutoModelForCausalLM),
    # ############################# Meta LLama Models ############################# #
    'CodeLlama-7b-Python-hf': (AutoConfig, AutoModelForCausalLM),
    'CodeLlama-13b-Python-hf': (AutoConfig, AutoModelForCausalLM),
    'CodeLlama-34b-Python-hf': (AutoConfig, AutoModelForCausalLM),
    'HF-Llama-3-8B-Instruct': (AutoConfig, AutoModelForCausalLM),
    # ############################# DeepSeek-Coder Models ############################# #
    'deepseek-coder-1.3b-base': (AutoConfig, AutoModelForCausalLM),
    'deepseek-coder-1.3b-instruct': (AutoConfig, AutoModelForCausalLM),
    'deepseek-coder-7b-instruct': (AutoConfig, AutoModelForCausalLM),
    'deepseek-coder-33b-instruct': (AutoConfig, AutoModelForCausalLM),
}

TOKENIZER_CLASSES = {
    'roberta': RobertaTokenizer,
    't5': T5Tokenizer,
    'codet5-small': RobertaTokenizer,
    'codet5-base': RobertaTokenizer,
    'codet5-large': RobertaTokenizer,
    # Official Documentation uses AutoTokenizer, but it is the same as RobertaTokenizer.
    # We want the same tokenization for all our models.
    'codet5-large-ntp-py': RobertaTokenizer,  # Official Documentation uses AutoTokenizer
    'bart': BartTokenizer,
    'gpt2': GPT2Tokenizer,
    'gpt2-xl': GPT2Tokenizer,
    'gpt-neo-125M': GPT2Tokenizer,
    'gpt-neo-1.3B': GPT2Tokenizer,
    'openai-gpt': OpenAIGPTTokenizer,
    'bert': BertTokenizer,
    'distilbert': DistilBertTokenizer,
}


def is_rank_0():
    return torch.distributed.get_rank() == 0


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def load_tokenizer(model_type):
    tokenizer_name = get_huggingface_path(model_type)

    if model_type in TOKENIZER_CLASSES:
        tokenizer_class = TOKENIZER_CLASSES[model_type]
    else:
        tokenizer_class = AutoTokenizer

    tokenizer = tokenizer_class.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Some Tokenizers do not have pad_token. We add it here. (It will only be used for ease of use in my pipeline.)
    if tokenizer.pad_token_id is None or tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Finish loading Tokenizer from %s", tokenizer_name)
    print("Finish loading Tokenizer from %s", tokenizer_name)
    return tokenizer


def load_base_model(model_type, config_name, model_path, load_in_8bit: bool = False):
    config_class, model_class = MODEL_CLASSES[model_type]

    config = config_class.from_pretrained(
        config_name if config_name else model_path,
        trust_remote_code=True,
        revision="main"
    )
    model = model_class.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision="main",
        device_map="auto",
        # torch_dtype=torch.bfloat16, #Adjust the percision accrording to your device
        ###########################
        torch_dtype=torch.float16,
        # load_in_8bit=load_in_8bit
    )

    logger.info("Finish loading Base model [%s] from %s", get_model_size(model), model_path)
    print("Finish loading Base model [%s] from %s", get_model_size(model), model_path)
    return config, model


def get_huggingface_path(model: str) -> str:
    # ############################# Microsoft Phi Models ############################## #
    if model == 'phi-3-mini-4k':
        huggingface_path = 'microsoft/Phi-3-mini-4k-instruct'  # 2.7B
    elif model == 'phi-3-small-8k':
        huggingface_path = 'microsoft/Phi-3-small-8k-instruct'  # 7B
    elif model == 'phi-3-medium-4k':
        huggingface_path = 'microsoft/Phi-3-medium-4k-instruct'  # 14B
    # ############################# Meta LLama Models ############################# #
    elif model == 'CodeLlama-7b-Python-hf':
        huggingface_path = 'codellama/CodeLlama-7b-Python-hf'
    elif model == 'CodeLlama-13b-Python-hf':
        huggingface_path = 'codellama/CodeLlama-13b-Python-hf'
    elif model == 'CodeLlama-34b-Python-hf':
        huggingface_path = 'codellama/CodeLlama-34b-Python-hf'
    elif model == 'HF-Llama-3-8B-Instruct':
        huggingface_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
    elif model == 'Meta-Llama-3-8B-Instruct':
        huggingface_path = '/home/abhinav/model_cache/Meta-Llama-3-8B-Instruct'
    elif model == 'Meta-Llama-3-70B-Instruct':
        huggingface_path = '/home/abhinav/model_cache/Meta-Llama-3-70B-Instruct'
    # ############################# DeepSeek-Coder Models ############################# #
    elif model == 'deepseek-coder-1.3b-base':
        huggingface_path = 'deepseek-ai/deepseek-coder-1.3b-base'
    elif model == 'deepseek-coder-1.3b-instruct':
        huggingface_path = 'deepseek-ai/deepseek-coder-1.3b-instruct'
    elif model == 'deepseek-coder-7b-instruct':
        huggingface_path = 'deepseek-ai/deepseek-coder-7b-instruct-v1.5'
    elif model == 'deepseek-coder-33b-instruct':
        huggingface_path = 'deepseek-ai/deepseek-coder-33b-instruct'
    else:
        raise NotImplementedError()

    return huggingface_path
