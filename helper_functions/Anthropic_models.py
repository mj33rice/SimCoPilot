import anthropic
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    
    )
def anthropic_models_gen(model, message, program_type, comment_symbol, max_gen_tokens):

    message = client.messages.create(
        model = model,
        system = f"{program_type} code generation", 
        max_tokens=max_gen_tokens,
        temperature=0,
        messages = message
    )
    return(message.content[0].text)