import torch
import tiktoken
from .xformer import get_huggingface_path, load_base_model, load_tokenizer

model_context = {
    #OpenAI GPT models
    "gpt-3.5-turbo-0125": {
        "context": 16385,
        "max_out": 4096
    },
    "gpt-4-turbo": {
        "context": 128000,
        "max_out": 4096
    },
    #Anthropic models
    "claude-3-opus-20240229": {
        "context": 200000,
    },
    "claude-3-sonnet-20240229": {
        "context": 200000,
    },
    "claude-3-haiku-20240307": {
        "context": 200000,
    },
    #HF Models
    "deepseek-coder-1.3b-base": {
        "context": 1024, #16384
        "max_out": 4096,
        "bos_token_id": 32013,
        "eos_token_id": 32014,
    },
    "deepseek-coder-1.3b-instruct": {
        "context": 4096, #16384
        "max_out": 4096,
        "bos_token_id": 32013,
        "eos_token_id": 32014,
    },
}

# Specify to use the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def num_tokens_from_HF_models(text, model):
    tokenizer = load_tokenizer(model)
    num_tokens = 0
    for message in text:
        # Ensure the message content is a string
        content = str(message['content'])
        num_tokens += len(tokenizer.tokenize(content))
    return num_tokens

def get_response_from_HF_models(messages, model, max_new_tokens=200, do_sample=False, 
                                top_k=50, top_p=0.95, num_return_sequences=1):
    config_name = None  # or replace with the actual config name if available
    model_path = get_huggingface_path(model)  # assuming get_huggingface_path is available in the current scope
    tokenizer = load_tokenizer(model)
    config, gen_model = load_base_model(model, config_name, model_path)
    
    # Move the model to the device
    gen_model = gen_model.to(device)
    
    # Prepare the inputs and move them to the device
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(device)
    
    # tokenizer.eos_token_id is the id of <|EOT|> token
    outputs = gen_model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p, 
                             num_return_sequences=num_return_sequences, eos_token_id=tokenizer.eos_token_id)
    return(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))

def truncate_input(before, after, token_len_to_remove, model, max_gen_tokens, keep_window=None, gpt_models_list = ['gpt-3.5-turbo-0125', 'gpt-4-turbo'], tokenizer=None):
    # Set the default keep_window if it's not provided
    if keep_window is None:
        keep_window = max_gen_tokens

    if model in gpt_models_list:
        tokenizer = tiktoken.encoding_for_model(model)
        before_tokens = tokenizer.encode(before)
        after_tokens = tokenizer.encode(after)
        
    else:
        #HF tokenizer load the tokenizer
        tokenizer = load_tokenizer(model)
        # Tokenize the 'before' and 'after' sections
        before_tokens = tokenizer.encode(before)
        after_tokens = tokenizer.encode(after)
    
    # Truncate the 'after' section from the tail if 'after' is available
    if len(after)>0 and token_len_to_remove > 0:
        if len(after_tokens) > keep_window:
            remove_len = min(len(after_tokens) - keep_window, token_len_to_remove)
            after_tokens = after_tokens[:-remove_len]
            token_len_to_remove -= remove_len

    # Truncate the 'before' section from the head
    if token_len_to_remove > 0 and len(before_tokens) > keep_window:
        remove_len = min(len(before_tokens) - keep_window, token_len_to_remove)
        before_tokens = before_tokens[remove_len:]
        token_len_to_remove -= remove_len
    return tokenizer.decode(before_tokens), tokenizer.decode(after_tokens)
