# SimCoPilot: Evaluating Models for Co-pilot-Style Code Generation
SimCoPilot is a benchmark for evaluating LLMs as "copilot"-style interactive coding assistants, testing their ability to integrate and complete code within complex real-world software environments.

![Figure 1: Workflow for each of the 1,163 programming tasks in SIMCOPILOT.](figures/Workflow.png "Figure 1: Workflow for each of the 1,163 programming tasks in SIMCOPILOT.")
## Dataset


The data for this project can be found in the ` dataset/SimCoPilot.csv.zip` file. 

**Hosting, Licensing, and Maintenance Plan.**
- **Dataset and Metadata Access.** The dataset and its associated metadata, documented using the Croissant metadata framework, can be viewed and downloaded at [https://huggingface.co/datasets/mj33/SimCoPilot](https://huggingface.co/datasets/mj33/SimCoPilot).
- **Licensing:** The data is shared under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) and code is licensed under MIT License.
- **Maintenance Plan:** We commit to maintaining the dataset with regular updates and revisions to correct any issues and integrate new contributions. Updates will be documented in the repository's release notes section.

## 🚀 Getting Started

1. Clone the repository.
2. Install necessary dependencies.
3. Run the analysis on your target codebase by specifying the file path and dependency range parameters.

To install the Dependency Analyzer, clone this repository and run the setup script:

```bash
git clone https://github.com/mj33rice/SimCoPilot.git
pip install -r requirements.txt
```

<details>
<summary>OpenAI & Anthropic modles setup</summary>
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
</details> 


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
<details>
<summary>Run from the Bash Script</summary>

```bash
#For Python tasks
chmod +x run_python_paral.sh
./run_python_paral.sh

#For Java tasks
chmod +x run_java_paral.sh
./run_java_paral.sh
```
</details>

# Post Processing
```python
python -m helper_functions.update_post_process_and_eval ./PATH/to/result_folder
```

## Example of Post-Processing

For detailed examples of code Post-Processing, please refer to the [Example Code Analysis Demo](./example_code/README.md)

## Contributions

Contributions to enhance the tool's capabilities, including support for more languages or improved analysis algorithms, are welcome. Please submit a pull request or an issue for discussion.

## License

The data is shared under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) and code is licensed under MIT License.
