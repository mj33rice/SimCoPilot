import os
from .xformer import get_huggingface_path, load_base_model, load_tokenizer

model_context = {
    "gpt-3.5-turbo-0125": {
        "context": 16385,
        "max_out": 4096
    },
    "gpt-4-turbo": {
        "context": 128000,
        "max_out": 4096
    },
    "deepseek-coder-1.3b-instruct": {
        "context": 4096,
        "max_out": 4096,
        "bos_token_id": 32013,
        "eos_token_id": 32014,
    },
    "deepseek-coder-7b-instruct": {
        "context": 4096,
        "max_out": 4096,
        "bos_token_id": 100000,
        "eos_token_id": 100015,
    },
    "deepseek-coder-33b-instruct": {
        "context": 4096,
        "max_out": 16384,
        "bos_token_id": 32013,
        "eos_token_id": 32021,
    },
    "phi-3-mini-4k": {
        "context": 4096,
        "max_out": 4096,
        "bos_token_id": 1,
        "eos_token_id": 32000,
    },
    "phi-3-small-8k": {
        "context": 4096,
        "max_out": 8192,
        "bos_token_id": 100257,
        "eos_token_id": 100257,
    },
    "phi-3-medium-4k": {
        "context": 4096,
        "max_out": 4096,
        "bos_token_id": 1,
        "eos_token_id": 32000,
    },
    "HF-Llama-3-8B-Instruct": {
        "context": 4096,
        "max_out": 8192,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
    },
    "Meta-Llama-3-8B-Instruct": {
        "context": 4096,
        "max_out": 8192,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
    },
    "Meta-Llama-3-70B-Instruct": {
        "context": 4096,
        "max_out": 8192,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
    },
}


def num_tokens_from_HF_models(text, tokenizer):
    num_tokens = 0
    for message in text:
        # Ensure the message content is a string
        content = str(message['content'])
        num_tokens += len(tokenizer.tokenize(content))
    return num_tokens


def load_HF_model(model_name: str):
    
    config_name = None  # or replace with the actual config name if available
    model_path = get_huggingface_path(model_name)  # assuming get_huggingface_path is available in the current scope
    tokenizer = load_tokenizer(model_name)
    config, model = load_base_model(model_name, config_name, model_path)

    # # Specify to use the GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # # Move the model to the device
    # model = model.to(device)

    return model, tokenizer


def load_Meta_model(model_name: str):
    from llama import Llama
    model = Llama.build(
        ckpt_dir=get_huggingface_path(model_name),
        tokenizer_path=os.path.join(get_huggingface_path(model_name), "tokenizer.model"),
        max_seq_len=model_context[model_name]["context"],
        max_batch_size=1,
    )
    tokenizer = model.tokenizer
    return model, tokenizer

def get_response_from_HF_models(
        messages, gen_model, tokenizer, max_new_tokens=200, do_sample=False,
        top_k=50, top_p=0.95, num_return_sequences=1
):

    # Prepare the inputs and move them to the device
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(gen_model.device)
    
    # tokenizer.eos_token_id is the id of <|EOT|> token
    outputs = gen_model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_k=top_k if do_sample else None,
        top_p=top_p if do_sample else None,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


def get_response_from_Meta_models(
        messages, gen_model, tokenizer, max_new_tokens=200, do_sample=False,
        top_k=50, top_p=0.95, num_return_sequences=1
):
    # Prepare the inputs and move them to the device
    prompt_tokens = [gen_model.formatter.encode_dialog_prompt(messages)]
    outputs, generation_logprobs = gen_model.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=max_new_tokens,
        temperature=0.0,  # For deterministic generation
        top_p=top_p,
        logprobs=False,
        echo=False,  # Do not echo the prompt in the response
    )
    return tokenizer.decode(outputs[0])


def truncate_input(before, after, token_len_to_remove, tokenizer, max_gen_tokens, keep_window=None):
    # Set the default keep_window if it's not provided
    if keep_window is None:
        keep_window = max_gen_tokens

    # Tokenize the 'before' and 'after' sections
    before_tokens = tokenizer.encode(before)
    after_tokens = tokenizer.encode(after)
    
    # Truncate the 'after' section from the tail if 'after' is available
    if len(after) > 0 and token_len_to_remove > 0:
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


def truncate_input_for_Meta(before, after, token_len_to_remove, tokenizer, max_gen_tokens, keep_window=None):
    # Set the default keep_window if it's not provided
    if keep_window is None:
        keep_window = max_gen_tokens
    
    # Tokenize the 'before' and 'after' sections
    before_tokens = tokenizer.encode(before, bos=False, eos=False)
    after_tokens = tokenizer.encode(after, bos=False, eos=False)
    
    # Truncate the 'after' section from the tail if 'after' is available
    if len(after) > 0 and token_len_to_remove > 0:
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
