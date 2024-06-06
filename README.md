# SimCoPilot
A framework to evaluate LLMs' capability in source code completion. To thoroughly assess the performance of the Language Models in code generation
tasks, we propose establishing various "evaluation checkpoints". At these checkpoints, the model’s
output will be assessed based on how well it continues or complements the patterns established by
the preceding code or achieves the functionality specified by the predefined criteria.

# Dependency Analyzer

In addition to the initial evaluation metrics, we plan to refine our assessment by categorizing the
"evaluation checkpoints" based on the length of code dependencies and logic component of the
to-complete code.

## Features

- **Dependency Length Analysis**: Evaluates dependencies based on the length of code preceding a given point, quantifying the distance between the definitions of variables, functions, or classes and their respective calls or implementations in the subsequent sections of code, quantified by the number of lines.
- **Reason and Horizon Categories**: Introduces two crucial analysis categories to better assess
dependency impact and logic handling:
  - **Horizon Category**: Assesses dependency lengths ranging from "Short-Range" to "Cross-Module".
  - **Reason Category**: Evaluates conditional logic, loop terminations, pattern usage, and context awareness.

These categories will offer a comprehensive
analysis of the Language Models’ code synthesis capabilities. Here are some potential categories to
consider:

### Horizon Category

| Horizon Category | Definition | Characteristics and Examples |
| ---------------- | ---------- | ---------------------------- |
| **Short-Range** | Involves dependencies within a few lines of code. | Local variables, immediate function calls. |
| **Medium-Range** | Covers dependencies over a moderate number of lines in the same file. | References to class variables, methods defined earlier. |
| **Long-Range** | Dependencies extend over a large portion of code. | Understanding of the project structure, including distant components. |
| **Cross-Module** | Relies on elements defined in different modules or libraries. | Requires understanding of external dependencies and library APIs. |


### Reason Category

| Reason Category | Definition | Characteristics and Examples |
| --------------- | ---------- | ---------------------------- |
| **If-else Reasoning** | Involves understanding the logic body of if and else statements. | Assists in developing coherent logic flow in conditional structures. E.g., Suggesting complementary else logic based on the conditions specified in the if statement, or recommending elif branches for multiple conditions. |
| **Define Stop Criteria** | Involves creating the stop condition for loops based on preceding code. | Analyzes the code to determine loop termination conditions. E.g., Suggesting a loop’s stop condition based on the initialization and usage of loop variables |
| **Pattern-Based Completion** | Completing code based on recognized coding patterns or best practices. | Recognizes and suggests implementation based on established patterns. |

## Usage

1. Clone the repository.
2. Install necessary dependencies.
3. Run the analysis on your target codebase by specifying the file path and dependency range parameters.

To install the Dependency Analyzer, clone this repository and run the setup script:

```bash
git clone https://github.com/mj33rice/LLMs_Program_Synthesis_Eval.git
pip install -r requirements.txt
```

## OpenAI & Anthropic modles setup
1. Install the necessary Python packages:
```bash
pip install anthropic
```
2. Open your terminal and type the following command:
```bash
nano ~/.bash_profile 
```
If you’re using a newer version of macOS, you might need to use `~/.zshrc` instead:
(or nano ~/.zshrc if you’re using a newer version of macOS)
```bash
nano ~/.zshrc
```
3. Add the following line to the file, replacing `your-api-key-here` with your actual API key:
```bash
 export ANTHROPIC_API_KEY='your-api-key-here' 
```
 If you're using OpenAI, use this line instead:
```bash
 export OPENAI_API_KEY='your-api-key-here'
```
4. Save the file and exit the editor (press `Ctrl+O`, then `Enter`, then `Ctrl+X`)
5. Load the updated profile by running: 

```bash
source ~/.bash_profile (or source ~/.zshrc)
```

## Run from the Bash Script
```bash
#For Python tasks
chmod +x run_python_paral.sh
./run_python_paral.sh

#For Java tasks
chmod +x run_java_paral.sh
./run_java_paral.sh
```

## How to Run 
```python
# LLMs Generations
CUDA_VISIBLE_DEVICES=2 python analyzer_demo_input_LNs.py ./example_code/Python/simplex_method/simplex_method.py ./example_code/Python/simplex_method/simplex_method.json --read_dependency_results --update_def_line --gen_model deepseek-coder-1.3b-instruct --code_gen_mode with_afterlines 
```
In the above command:

`--gen_model`: `deepseek-coder-1.3b-instruct`: This argument specifies the model to be used for code generation. In this context, it is used to select the 'deepseek-coder-1.3b-instruct' model for generating code.

`--code_gen_mode`: `with_afterlines`: This argument determines the mode of code generation. The 'with_afterlines' mode likely controls whether or not additional lines of code are generated after the main code block.
Debug Mode

`--code_gen_timeout`: This argument sets a timeout limit for the code generation process. It is used in the LLMs_gen_and_post_process function to prevent the code generation process from running indefinitely. 

# Debug Mode

```python
CUDA_VISIBLE_DEVICES=2 python analyzer_demo_input_LNs.py ./example_code/Python/simplex_method/simplex_method.py ./example_code/Python/simplex_method/simplex_method.json --read_dependency_results --update_def_line --gen_model deepseek-coder-1.3b-instruct --code_gen_mode with_afterlines --skip_save_csv
```
# Post Processing
```python
python -m helper_functions.update_post_process_and_eval ./Analysis_Results/Post_process/Check_pass_hist/no_instructions/Python/test_update/Image_Filtering/ > ./Analysis_Results/Post_process/Check_pass_hist/no_instructions/Python/test_update/stdout_post_process_Image_Filtering_May_23th 2> ./Analysis_Results/Post_process/Check_pass_hist/no_instructions/Python/test_update/stderr_post_process_Image_Filtering_May_23th&
```

## Example Code Analysis

For detailed examples of code analysis, please refer to the [Example Code Analysis Demo](./example_code/EXAMPLE.md).

## Contributions

Contributions to enhance the tool's capabilities, including support for more languages or improved analysis algorithms, are welcome. Please submit a pull request or an issue for discussion.

## License

This project is licensed under [SPECIFY LICENSE] - see the LICENSE file for details.
