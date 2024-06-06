import tiktoken
import signal
import time
from openai import OpenAI
from difflib import SequenceMatcher
from transformers import pipeline
from .post_gen_process import clean_code, auto_indent, auto_bracket_matcher, trim_similar_edges
# from .xformer import get_huggingface_path, load_base_model, load_tokenizer
from .HF_models import num_tokens_from_HF_models, get_response_from_HF_models, model_context, truncate_input
# import torch
from .Anthropic_models import anthropic_models_gen

client = OpenAI()
anthropic_models_list = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
gpt_models_list = ['gpt-3.5-turbo-0125', 'gpt-4-turbo']

# Helper function to handle the timeout
class TimeoutException(Exception):
    pass  # Simple exception to be raised on timeout

def raise_timeout(signum, frame):
    raise TimeoutException()

def get_model_info(model, model_context):
    try:
        max_tokens = model_context[model]["context"]
        return max_tokens
    except Exception as e:
        print("Failed to retrieve model information:", str(e))
        return None
    
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-4-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def remove_repeated_lines(between, before_lines, after_lines):
    """
    Removes lines in 'between' that are exactly present in 'before_lines' or 'after_lines'.

    Args:
    between (str): The 'between' part of the code.
    before_lines (list[str]): Lines of code before the 'between' part.
    after_lines (list[str]): Lines of code after the 'between' part.

    Returns:
    str: The 'between' part of the code with exact duplicate lines removed.
    """
    before_lines_set = set(before_lines)
    after_lines_set = set(after_lines)
    between_lines = between.split('\n')
    
    # Keep only lines that are not in before_lines_set or after_lines_set
    unique_lines = [line for line in between_lines if line not in before_lines_set and line not in after_lines_set]
    
    return '\n'.join(unique_lines)

def extract_code_results(gen_code_dict, last_post_process_step=None):
    """
    Extracts the code results from the generation and post-processing steps, returning both
    the step and the code result.

    Args:
        gen_code_dict (dict): Dictionary containing the results of the code generation and post-processing.
        last_post_process_step (list of str, optional): Specific keys to extract from gen_code_dict in order.
            If None, extracts the last non-empty result from the predefined keys.

    Returns:
        tuple: A tuple containing the step name and the result code from the last specified or non-empty processing step.
    """
    default_steps = ['original_code', 'cleaned_code', 'trimmed_code', 'indented_code']
    
    if last_post_process_step is None:
        # Loop through the default list in reverse to find the last non-empty result
        for step in reversed(default_steps):
            if gen_code_dict[step]:
                return step, gen_code_dict[step]
    else:
        # Loop through the provided list to extract the result of the last specified step
        for step in last_post_process_step:
            if step in gen_code_dict and gen_code_dict[step]:
                return step, gen_code_dict[step]
    
    return "", ""  # Return empty strings if no valid code or step was found

def read_prompt_from_file(file_path, structured_context, num_pairs, rounds):
    with open(file_path, 'r') as file:
        content = file.read()
    return content.format (structured_context=structured_context, num_pairs=num_pairs, rounds=rounds)

def generate_code_with_no_instruction(before_lines, after_lines, model, program_type):
    before = "\n".join(before_lines)
    after = "\n".join(after_lines)

    # Providing a brief description of what the missing code should do
    task_description = f"# Description: Fill in the missing {program_type} code that logically connects the 'before' code block to the 'after' code block. The generated code should perform a distinct functionality without duplicating or repeating the logic present in either the 'before' or 'after' code blocks."

    if program_type == "Python":
        # First set of examples
        example1_before = "def calculate_area(length, width):\n    # Code for calculating area"
        example1_after = "    return area\n\nprint(calculate_area(5, 3))"
        example1_generated = "    area = length * width"
        example1_description = "# Calculate the area of a rectangle given length and width."

        # Second, more detailed set of examples
        example2_before = (
            "def process_data(data):\n"
            "    # Code to process data\n"
            "    processed_data = None\n"
            "    # Additional processing steps"
        )
        example2_after = (
            "    return processed_data\n\n"
            "data = [1, 2, 3, 4]\n"
            "print(process_data(data))"
        )
        example2_generated = (
            "    processed_data = [d * 2 for d in data]\n"
            "    # Additional operations on processed_data"
        )
        example2_description = "# Process the data by doubling each element and then perform additional operations."
    elif program_type == "Java":
        # First set of examples
        example1_before = "public class AreaCalculator {\n    public static double calculateArea(double length, double width) {"
        example1_after = "        return area;\n    }\n\n    public static void main(String[] args) { System.out.println(calculateArea(5, 3)); }"
        example1_generated = "        double area = length * width;"
        example1_description = "// Calculate the area of a rectangle given length and width."

        # Second, more detailed set of examples
        example2_before = (
            "public class DataProcessor {\n    public static List<Integer> processData(List<Integer> data) {"
            "        List<Integer> processedData = new ArrayList<>();\n        // Additional processing steps"
        )
        example2_after = (
            "        return processedData;\n    }\n\n    public static void main(String[] args) {"
            "        List<Integer> data = Arrays.asList(1, 2, 3, 4);\n        System.out.println(processData(data)); }"
        )
        example2_generated = (
            "        for (int d : data) { processedData.add(d * 2); }\n        // Additional operations on processedData"
        )
        example2_description = "// Process the data by doubling each element and then perform additional operations."
    else:
        raise ValueError("Unsupported program type. Only 'Python' and 'Java' are supported.")

    prompt = (
        f"As a senior software developer, you are tasked with generating {program_type} code that connects two parts of a larger codebase. "
        f"The generated code should strictly adhere to {program_type} syntax, including correct use of indentation. "
        f"Furthermore, it is crucial that this code provides unique functionality and does not replicate, either in logic or functionality, any part of the 'before' or 'after' sections provided. "
        f"Your goal is to create a seamless and syntactically correct transition between these two parts, ensuring that the generated code performs a unique task as described.\n\n"
        f"Example 1:\n{example1_before}\n{example1_description}\n# --BEGIN MISSING CODE--\n{example1_generated}\n# --END MISSING CODE--\n{example1_after}\n\n"
        f"Example 2:\n{example2_before}\n{example2_description}\n# --BEGIN MISSING CODE--\n{example2_generated}\n# --END MISSING CODE--\n{example2_after}\n\n"
        f"{task_description}\nYour Task:\n{before}\n"
        "# --BEGIN MISSING CODE--\n"
        f"# [Ensure your generated code here follows {program_type} syntax and does not duplicate functionality from before or after sections. Do not generate any contect that is not code syntax.]\n"
        "# --END MISSING CODE--\n"
        f"{after}"
    )
    try:
        input_messages = [   {"role": "system", "content": f"{program_type} code generation"},
                            {"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            # model= "gpt-4-0125-preview", #gpt-4-1106-preview #gpt-4 #gpt-4-32k #gpt-4-0613 #gpt-3.5-turbo-0613 #gpt-4-0125-preview
            model = model,
            messages=input_messages,
        )

        generated_code = response.choices[0].message.content

        # # Post-processing: Remove any non-code text (lines not starting with whitespace or 'def')
        # code_lines = [line for line in generated_code.split('\n') if line.strip() and (line.startswith('    ') or line.startswith('def'))]
        # between = '\n'.join(code_lines)
        between = generated_code

        # Check for similarity with before, after parts, and existing logic, reject if too similar
        similarity_checks = [
            SequenceMatcher(None, before, between).ratio(),
            SequenceMatcher(None, after, between).ratio(),
            # SequenceMatcher(None, existing_logic_str, between).ratio()
        ]
        if any(similarity > 0.75 for similarity in similarity_checks):
            between = "Generated code is too similar to the existing code. Please try again."

    except Exception as e:
        between = f"An error occurred: {e}"
    return between


def create_input_messages(program_type, prompt_template, before, after, examples_content, instructions_in_comment, comment_symbol):
    # Optionally include the instruction based on the last comment in the 'before' part
    instruction_text = " Generate code based on the instructions in the last comment in the 'before' part. " if instructions_in_comment else ""

    # Add instruction about indentation for Python
    if program_type == "Python":
        instruction_text += "\n\nNOTE: Python uses indentation to define blocks of code. Therefore, it's crucial to use the correct indentation when generating your code."
        if after:
            instruction_text += " Ensure your code connects seamlessly with the 'after' section using the correct indentation."
        else:
            instruction_text += " Consider the indentation of the 'before' section when generating your code."
    
    # Format the prompt with the necessary sections
    formatted_after_section = f"After Section:\n{after}" if after else ""
    
    prompt = prompt_template.format(
        program_type=program_type,
        instruction_text=instruction_text,
        before_section=before,
        begin_missing_code=f"{comment_symbol} --BEGIN MISSING CODE--",
        end_missing_code=f"{comment_symbol} --END MISSING CODE--",
        after_section=formatted_after_section,
        examples=examples_content  # This will include the examples directly in the prompt if provided
    )    
    input_messages = [
        {"role": "system", "content": f"{program_type} code generation"},
        {"role": "user", "content": prompt}
    ]
    return input_messages

def create_anthropic_input_messages(program_type, prompt_template, before, after, examples_content, instructions_in_comment, comment_symbol):
    # Optionally include the instruction based on the last comment in the 'before' part
    instruction_text = " Generate code based on the instructions in the last comment in the 'before' part. " if instructions_in_comment else ""
    # Add instruction about indentation for Python
    Python_instruction = "- It must use correct Python indentation" if program_type == "Python" else ""
    prompt = prompt_template.format(
        program_type=program_type,
        instruction_text=instruction_text,
        before_section=before,
        after_section=after,
        begin_missing_code=f"{comment_symbol} --BEGIN MISSING CODE--",
        end_missing_code=f"{comment_symbol} --END MISSING CODE--",
        Python_instruction=Python_instruction,
        examples=examples_content  # This will include the examples directly in the prompt if provided
    )
    input_messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
    return input_messages


def get_prompt(model, after_lines, prompts_path, program_type, with_examples):
    if model in anthropic_models_list:
        prompts_path = f"{prompts_path}/Anthropic"

    if after_lines:
        prompt_file_path = f"{prompts_path}/before_after_template.txt"
        examples_file_path = f"{prompts_path}/{program_type.lower()}_examples_with_before_after.txt" if with_examples else None
    else:
        prompt_file_path = f"{prompts_path}/without_after_template.txt"
        examples_file_path = f"{prompts_path}/{program_type.lower()}_examples_without_after.txt" if with_examples else None

    return prompt_file_path, examples_file_path

#Add HF model support generate_code_strict_format()
##############################################################################################################
def generate_code_strict_format(before_lines, after_lines, model, program_type, 
                                prompts_path="./helper_functions/prompt_templates", 
                                instructions_in_comment=False, with_examples=False):

    before = "\n".join(before_lines)
    after = "\n".join(after_lines)
    
    comment_symbol = "//" if program_type == "Java" else "#"

    # Choose the prompt and examples files based on whether after_lines is provided
    prompt_file_path, examples_file_path = get_prompt(model, after_lines, prompts_path, program_type, with_examples)
    # Read the prompt from the chosen file
    with open(prompt_file_path, 'r') as file:
        prompt_template = file.read()

    # Optionally add examples to the prompt
    examples_content = ""
    if with_examples and examples_file_path:
        with open(examples_file_path, 'r') as file:
            examples_content = file.read()
    
    model_max_tokens = get_model_info(model, model_context)
    # print("Maximum token limit for the model:", model_max_tokens)
    
    if model in anthropic_models_list:
        # No truncation to claude-3 model since it has 200k tokens limit & no public tokenizer available
        input_messages = create_anthropic_input_messages(program_type, prompt_template, before, after, examples_content, instructions_in_comment, comment_symbol)
        max_gen_tokens = 200
    else:
        input_messages = create_input_messages(program_type, prompt_template, before, after, examples_content, instructions_in_comment, comment_symbol)
        max_gen_tokens = 200
        if model in gpt_models_list:
            num_tokens = num_tokens_from_messages(input_messages, model)
        else:
            try:
                num_tokens = num_tokens_from_HF_models(input_messages, model)
            except Exception as e:
                print("Failed to calculate the number of tokens:", str(e))
                num_tokens = 0
        # Calculate the maximum token limit for the "before" and "after" sections
        token_len_to_remove = -(model_max_tokens - max_gen_tokens - num_tokens)
        if token_len_to_remove > 0:
            # Truncate the input messages if the token length exceeds the limit
            # First truncate the 'before' section and keep at least keep_window tokens from the head
            # Then truncate the 'after' section and keep at least keep_window tokens from the tail
            before, after = truncate_input(before, after, token_len_to_remove, model, max_gen_tokens)
            input_messages = create_input_messages(program_type, prompt_template, before, after, examples_content, instructions_in_comment, comment_symbol)
    try:
        if model in gpt_models_list:
            response = client.chat.completions.create(
                                                        model=model,
                                                        messages=input_messages,
                                                        max_tokens=max_gen_tokens,  # Adjust based on the typical length of the missing code, considering examples
                                                        temperature=0, # Set to 0 for deterministic results
                                                        stop=[f"{comment_symbol} --END MISSING CODE--"]  # Ensures model stops generating code at this marker
                                                    )
            generated_code = response.choices[0].message.content
        elif model in anthropic_models_list:
            generated_code = anthropic_models_gen(model, input_messages, program_type, comment_symbol, max_gen_tokens)
        else:
            generated_code = get_response_from_HF_models(input_messages, model, max_new_tokens=max_gen_tokens)

        # Extract the generated code between the designated markers
        start_marker = f"{comment_symbol} --BEGIN MISSING CODE--"
        end_marker = f"{comment_symbol} --END MISSING CODE--"
        start_idx = generated_code.find(start_marker) + len(start_marker)
        end_idx = generated_code.find(end_marker)

        if start_idx == len(start_marker) - 1 or end_idx == -1:
            # If markers are not found, return the complete generated output
            code_snippet = generated_code
        else:
            code_snippet = generated_code[start_idx:end_idx].strip()

    except Exception as e:
        code_snippet = f"An error occurred: {e}"

    return code_snippet

def post_process_gen_code(gen_code, before_lines, after_lines, program_type, 
                                clean_gen_code=True, remove_gen_repeated_lines=True, 
                                add_auto_indent=True, print_post_process=False):
    # Initialize the results dictionary with empty values for each step
    results = {
        'original_code': '',
        'cleaned_code': '',
        'trimmed_code': '',
        'indented_code': ''
    }
    post_process_steps = [] # List to store the post-processing steps applied
    tab_indent = 4
    # Dictionary to store results after each step
    results['original_code'] = gen_code
    if print_post_process:
        print('-' * 80)
        print("Original Code:\n")
        print(gen_code.replace('\t', ' ' * tab_indent))

    # Clean generated code
    if clean_gen_code:
        post_process_steps.append('cleaned_code')
        cleaned_code = clean_code(gen_code, program_type)
        results['cleaned_code'] = cleaned_code
        gen_code = cleaned_code  # Update gen_code to the cleaned version
        if print_post_process:
            print('-' * 80)
            print("Cleaned Code:\n")
            print(cleaned_code.replace('\t', ' ' * tab_indent))

    # Remove repeated lines
    if remove_gen_repeated_lines:
        post_process_steps.append('trimmed_code')
        trimmed_code = trim_similar_edges(gen_code, before_lines, after_lines, program_type)
        results['trimmed_code'] = trimmed_code
        gen_code = trimmed_code  # Update gen_code to the trimmed version
        if print_post_process:
            print('-' * 80)
            print("Trimmed Code:\n")
            print(trimmed_code.replace('\t', ' ' * tab_indent))

    # Auto-indent code (specifically for Python)
    if add_auto_indent:
        post_process_steps.append('indented_code')
        if program_type == "Java":
            indented_code = auto_bracket_matcher(before_lines, gen_code, after_lines)
        elif program_type == "Python":
            indented_code = auto_indent(before_lines, gen_code, after_lines)
        else:
            raise ValueError("Auto-indentation is only supported for Python and Java programs.")
        results['indented_code'] = indented_code
        gen_code = indented_code  # Update gen_code to the indented version
        if print_post_process:
            print('-' * 80)
            # print(f"Auto Indented Code:\n{gen_code}\n")
            print("Auto Indented Code:\n")
            print(indented_code.replace('\t', ' ' * tab_indent))

    # Return the dictionary containing all intermediate results
    return results, post_process_steps

def LLMs_gen_and_post_process(before_lines, after_lines, program_type, 
                                clean_gen_code=True, remove_gen_repeated_lines=True, 
                                add_auto_indent=True, print_post_process=True, 
                                gen_mode='no_afterlines', model='gpt-4-0125-preview',
                                gen_time_out=None):
    # Wrap the code generation step with signal for timeout handling
    try:
        if gen_time_out:
            signal.signal(signal.SIGALRM, raise_timeout)
            signal.alarm(gen_time_out)  # Set the alarm

        # Generate initial code based on the mode
        if gen_mode == 'no_afterlines':
            after_lines = ''
            gen_code = generate_code_strict_format(before_lines, after_lines, model, program_type)
        elif gen_mode == 'no_instruction':
            gen_code = generate_code_with_no_instruction(before_lines, after_lines, model, program_type)
        elif gen_mode == 'with_afterlines':
            gen_code = generate_code_strict_format(before_lines, after_lines, model, program_type)
        else:
            raise ValueError(f"Invalid gen_mode: {gen_mode}")
        if gen_time_out:
            signal.alarm(0)  # Disable the alarm after successful generation
    except TimeoutException:
        return {'Error': 'Timeout: Code generation exceeded time limit'}, []
    except Exception as e:
        return {'Error': f"Unexpected error during code generation: {e}"}, []
    
    # Post-process the generated code
    results, post_process_steps = post_process_gen_code(gen_code, before_lines, after_lines, program_type, 
                                                        clean_gen_code, remove_gen_repeated_lines, 
                                                        add_auto_indent, print_post_process)

    return results, post_process_steps

def remove_non_code_text_enhanced(generated_code, program_type):
    lines = generated_code.split('\n')
    code_lines = []
    in_multiline_comment = False

    for line in lines:
        stripped_line = line.strip()
        if program_type == "Python":
            # Handle multi-line strings used as comments in Python
            if '"""' in stripped_line or "'''" in stripped_line:
                if stripped_line.count('"""') % 2 != 0 or stripped_line.count("'''") % 2 != 0:
                    in_multiline_comment = not in_multiline_comment
                # Consider case where multi-line string/comment starts and ends on the same line
                if stripped_line.count('"""') == 2 or stripped_line.count("'''") == 2:
                    in_multiline_comment = False
                continue  # Skip adding this line
            elif in_multiline_comment:
                continue
            elif stripped_line.startswith('#'):
                continue  # Skip single-line comments

        elif program_type == "Java":
            if in_multiline_comment:
                if stripped_line.endswith('*/'):
                    in_multiline_comment = False
                continue  # Skip lines until the end of the multi-line comment
            else:
                if stripped_line.startswith('/*'):
                    in_multiline_comment = True
                    continue  # Skip the start of a multi-line comment
                if stripped_line.startswith('//'):
                    continue  # Skip single-line comments

        # Add the line if it's not part of a comment
        if not stripped_line.startswith('//') and not in_multiline_comment:
            code_lines.append(line)

    return '\n'.join(code_lines)

# # Example usage
# generated_python_code = """
# '''
# This is a multi-line comment in Python
# '''
# def example_function():
#     # This is a single-line comment
#     print("Hello, World!")
# """

# generated_java_code = """
# /**
#  * This is a multi-line comment in Java
#  */
# public class Example {
#     // This is a single-line comment
#     public static void main(String[] args) {
#         System.out.println("Hello, World!");
#     }
# }
# """

# print(remove_non_code_text_enhanced(generated_python_code, "Python"))
# print(remove_non_code_text_enhanced(generated_java_code, "Java"))

