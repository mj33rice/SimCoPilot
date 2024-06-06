import os
from datetime import datetime
import argparse
import signal
import ast
import json
import pandas as pd
from termcolor import colored, cprint
from helper_functions.text_split_and_color import split_code, colored_line_numbers_and_text, add_line_numbers_and_color
from helper_functions.eval_code import load_test_cases_w_expected_output
from helper_functions.LLMs_gen  import LLMs_gen_and_post_process, extract_code_results, post_process_gen_code
import shutil
from multiprocessing import Pool


# from helper_functions.data_frame_operations import overwrite_columns, rename_file, overwrite_columns_in_all_files
# Get the current process id
pid = os.getpid()

def get_paths(code_task, task_path):
    # Load the paths from the JSON file
    with open(task_path) as f:
        paths = json.load(f)

    # Find the programming language based on the code_task
    for language, tasks in paths.items():
        if code_task in tasks:
            base_path = tasks[code_task]
            break
    else:
        raise ValueError(f'Unknown code task: {code_task}')

    test_cases_path = base_path + '.json'
    source_code_path = base_path + '.py' if language == 'Python' else base_path + '.java'

    return test_cases_path, source_code_path, language

def rename_file(tar_csv_file, new_folder_name):
    # Get the current time and format it as a string
    current_time = datetime.now().strftime('%m_%d_%H_%M')

    # Get the base name of the target file (without the extension)
    base_name = os.path.splitext(os.path.basename(tar_csv_file))[0]

    # Split the base name into parts by underscore
    parts = base_name.split('_')
    # Replace the last four parts with the current time
    parts[-4:] = current_time.split('_')
    # Join the parts back together to form the new base name
    new_base_name = '_'.join(parts)

    # Create the new file path by joining the directory with the new filename and the extension
    res_csv_file = os.path.join(new_folder_name, new_base_name + '.csv')

    return res_csv_file

def re_organize_folder(src_csv_folder, new_folder='pre_processed'):
    # Create the new folder if it doesn't exist
    new_folder_path = os.path.join(os.path.dirname(src_csv_folder), new_folder)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # Get the list of files in the source folder
    src_csv_files_list = os.listdir(src_csv_folder)

    # Move each file to the new folder
    for src_csv_file in src_csv_files_list:
        src_file_path = os.path.join(src_csv_folder, src_csv_file)
        dst_file_path = os.path.join(new_folder_path, src_csv_file)
        shutil.move(src_file_path, dst_file_path)

def convert_to_float(val: str) -> float:
    try:
        return float(val)
    except ValueError:
        parts = val.split('/')
        if len(parts) == 2:
            num, denom = parts
            if denom != '0':
                return float(num) / float(denom)
        return 0.0
def res_check(df, column):
    total = df[column].apply(convert_to_float).notna().sum()
    passed = (df[column].apply(convert_to_float) == 1).sum()
    ratio = passed / total * 100
    return f"{ratio:.2f}% ({passed}/{total})"

def get_output_to_check(code_task, language):
    # Load the output_to_check from the JSON file
    file_path = f"example_code/{language}/{language}_tasks_checkpoints.json"
    with open(file_path, 'r') as f:
        tasks = json.load(f)

    output_to_check = None
    for task in tasks:
        if task['Task_Name'] == code_task and 'output_to_check' in task:
            output_to_check = task['output_to_check']
            break

    return output_to_check

def process_and_evaluate_code(src_df, args, 
                              before_post_process_step='cleaned_code', steps=['original_code', 'cleaned_code', 'trimmed_code', 'indented_code'],
                              print_process=True):
    # Initialize the columns with default values
    src_df['post_process_eval_res'] = None
    src_df['post_process_pass_ratio'] = None
    src_df['2nd_post_process_steps'] = None

    # Get the 'code_task' value
    code_task = src_df["code_task"].values[0]
    # Get the paths
    test_cases_path, source_code_path, program_type = get_paths(code_task, args.task_path)
    # Get the output_to_check value
    output_to_check = get_output_to_check(code_task, program_type)

    for index, row in src_df.iterrows():
        # Load the 'original_code' from the 'gen_code_dict'
        gen_code_dict = row['gen_code_dict']
        original_gen_code = gen_code_dict['original_code']

        # Load the corresponding info from the row
        before = ast.literal_eval(row['before'])
        between = ast.literal_eval(row['between'])
        after = ast.literal_eval(row['after'])
        start_line = row['start_line']
        end_line = row['end_line']

        # # Skip the row if start_line is greater than end_line
        # if start_line > end_line:
        #     continue

        gen_code_pass_ratio = row['gen_code_pass_ratio']
        post_process_steps = row['gen_code_process_steps']

        # Check if gen_code_pass_ratio can be parsed into a Python object
        try:
            gen_code_pass_ratio_val = convert_to_float(gen_code_pass_ratio)
        except :
            # import pdb; pdb.set_trace()
            # If not, set it to a default value (e.g., 0)
            gen_code_pass_ratio_val = 0


        # If ast.literal_eval(gen_code_pass_ratio) != 1, then extract the steps after before_post_process_step in post_process_steps and their corresponding values from gen_code_dict
        if gen_code_pass_ratio_val != 1:
            start_index = steps.index(before_post_process_step) + 1
            prev_step = before_post_process_step
            for step in steps[start_index:]:
                if step in post_process_steps:
                    post_process_code_selected = gen_code_dict[step]
                    post_process_code_selected = post_process_code_selected.split('\n')
                    if print_process:
                        # Check if the previous step exists
                        if prev_step and prev_step in gen_code_dict:
                            print(colored(f"##############  Before Post Process ({prev_step})  ##############", "yellow"))
                            print(colored(gen_code_dict[prev_step],"cyan"))
                        print(colored("##############  Post-Process Step  ##############", "yellow"))
                        print(colored(step,"cyan"))
                        print(colored("##############  Code Display  ##############", "yellow"))
                        display_gen_code(before, between, after, post_process_code_selected, start_line, end_line)
                        # Use it as gen_code_selected to put into load_test_cases_w_expected_output()
                    post_process_eval_res, post_process_pass_ratio = load_test_cases_w_expected_output(test_cases_path, before, post_process_code_selected, after, source_code_path, output_to_check, eval_timeout=args.eval_timeout)

                    # Record the ast.literal_eval(load_test_cases_w_expected_output()) to "post_process_eval_res"; and corresponding key (e.g. indented_code) to "2nd_post_process_steps"
                    src_df.at[index, 'post_process_eval_res'] = post_process_eval_res
                    src_df.at[index, 'post_process_pass_ratio'] = post_process_pass_ratio
                    src_df.at[index, '2nd_post_process_steps'] = step
                    
                    # Update the previous step
                    prev_step = step
                    # We start from 'trimmed_code' if the result is not 1, then move on to indented_code
                    if convert_to_float(post_process_pass_ratio) == 1:
                        break
        else:
            # If gen_code_pass_ratio_val is already 1, copy the existing values to the post-process columns
            src_df.at[index, 'post_process_eval_res'] = src_df.at[index, 'gen_code_eval_res']
            src_df.at[index, 'post_process_pass_ratio'] = src_df.at[index, 'gen_code_pass_ratio']
            src_df.at[index, '2nd_post_process_steps'] = src_df.at[index, 'gen_code_process_steps']
    
    # Before returning the DataFrame, print the pass ratios
    print("gen_code_pass_ratio:", res_check(src_df, 'gen_code_pass_ratio'))
    print("post_process_pass_ratio:", res_check(src_df, 'post_process_pass_ratio'))
    return src_df

def display_gen_code(before, between, after, gen_code_selected, start_line, end_line):
    colored_before, colored_between, colored_after = colored_line_numbers_and_text(before, between, after, start_line, end_line)
    print(colored_before)
    colored_gen_code = add_line_numbers_and_color(gen_code_selected, start=start_line, text_color="red")
    print(colored("##############  Ground Truth  ##############", "yellow"))
    print(colored(colored_between,"cyan"))
    print(colored("############## Generated Code ##############", "yellow"))
    print(colored(colored_gen_code,"red"))
    print(colored("##############  Ground Truth  ##############", "yellow"))
    print(colored_after)

def main(args, src_csv_folder, output_file_path, gen_model, code_gen_mode='no_afterlines', short_range=5, medium_range=20, 
        eval_ground_truth=False, code_completion=True, eval_gen_code=True, get_task_labels=True, eval_post_process_code = True):
    src_csv_files_list = os.listdir(src_csv_folder)
    for src_csv_file in src_csv_files_list:
        if not src_csv_file.endswith('.csv'):
            continue
        src_csv_file_path = os.path.join(src_csv_folder, src_csv_file)
        src_df = pd.read_csv(src_csv_file_path)
        # Initialize an empty list to store the results
        results = []
        # Iterate over the rows in the DataFrame
        for index, row in src_df.iterrows():
            # Get the 'code_task' value
            code_task = src_df["code_task"].values[0]
            # Get the paths
            test_cases_path, source_code_path, program_type = get_paths(code_task, args.task_path)
            # Get the output_to_check value
            output_to_check = get_output_to_check(code_task, program_type)
            # Load the 'original_code' from the 'gen_code_dict'
            gen_code_dict = ast.literal_eval(row['gen_code_dict'])
            original_gen_code = gen_code_dict['original_code']

            # Load the corresponding info from the row
            before = ast.literal_eval(row['before'])
            between = ast.literal_eval(row['between'])
            after = ast.literal_eval(row['after'])
            start_line = row['start_line']
            end_line = row['end_line']

            # if start_line != 109 and end_line != 109:
            #     continue

            # Skip the row if start_line is greater than end_line
            if start_line > end_line:
                src_df = src_df.drop(index)
                continue

            
            # Run the post_process_gen_code() function
            gen_code_dict, post_process_steps = post_process_gen_code(original_gen_code, before, after, program_type, 
                                                                    clean_gen_code=True, remove_gen_repeated_lines=True, 
                                                                    add_auto_indent=True, print_post_process=True)
            
            # Extract the last non-empty result
            selected_last_post_process_step, gen_code_selected = extract_code_results(gen_code_dict, last_post_process_step=['cleaned_code'])
            gen_code_selected = gen_code_selected.split('\n')

            display_gen_code(before, between, after, gen_code_selected, start_line, end_line)

            if eval_gen_code:
                #Eval code with test cases:
                gen_code_eval_res, gen_code_pass_ratio  = load_test_cases_w_expected_output(test_cases_path, before, gen_code_selected, after, source_code_path, output_to_check, eval_timeout=args.eval_timeout)

            # Overwrite the specified columns in the DataFrame
            src_df.at[index, 'gen_code_dict'] = gen_code_dict
            src_df.at[index, 'gen_code_process_steps'] = post_process_steps
            src_df.at[index, 'selected_last_post_process_step'] = selected_last_post_process_step
            src_df.at[index, 'gen_code_selected'] = gen_code_selected
            src_df.at[index, 'gen_code_eval_res'] = str(gen_code_eval_res)
            src_df.at[index, 'gen_code_pass_ratio'] = gen_code_pass_ratio

        if eval_post_process_code:
            # Call the new function here
            src_df = process_and_evaluate_code(src_df, args)

        # Save the updated DataFrame to a new CSV file
        # src_df.to_csv('updated_test_df.csv', index=False)
        res_csv_file_path = os.path.join(src_csv_folder, rename_file(src_csv_file, 'Updated_post_process'))
        # Check if the directory exists
        res_dir = os.path.dirname(res_csv_file_path)
        if not os.path.exists(res_dir):
            # If not, create it
            os.makedirs(res_dir)
        # Save the DataFrame to the CSV file
        src_df.to_csv(res_csv_file_path, index=False)
        print(f"Columns overwritten successfully where matches found, and file {res_csv_file_path} updated.")
    
    # # After processing all files, re-organize the folder
    # re_organize_folder(src_csv_folder, new_folder='pre_processed')

if __name__ == "__main__":
    # try:
        parser = argparse.ArgumentParser(description='Analyze code dependencies.')
        parser.add_argument('csv_folder_path', type=str, help='Path to the csv folders')
        parser.add_argument('--task_path', type=str, default='./example_code/tasks_path.json', help='Path to the tasks folders')
        
        parser.add_argument('--output_file_path', type=str, default='./Analysis_Results/complete_output.csv', help='Analysis Results of Code Generation.')
        parser.add_argument('--short_range', type=int, default=10, help='Distance for Short-Range dependencies.')
        parser.add_argument('--medium_range', type=int, default=30, help='Distance for Medium-Range dependencies.')
        parser.add_argument('--gen_model', type=str, default='gpt-3.5-turbo-0125', help='LLM Selection for code generation task.')
        parser.add_argument('--code_gen_mode', type=str, default='with_afterlines', help='LLM mode for code generation task, such as no_afterlines, with_afterlines, no_instruction.')
        parser.add_argument('--eval_timeout', type=int, default=5*60, help='Timeout for evaluating the generated code.')
        parser.add_argument('--code_gen_timeout', type=int, default=3*60, help='Timeout for generating the code.')
        
        parser.add_argument('--read_dependency_results', action='store_true', default=False, help='Read dependency results from a CSV file.')
        parser.add_argument('--save_dependency_results', action='store_true', default=False, help='Save dependency results to a CSV file.')
        parser.add_argument('--update_def_line', action='store_true', default=True,  help='Update the definition line of variables when value reassign.')
        parser.add_argument('--show_task_specs', action='store_true', default=True, help='Show the task specs in the output.')
        
        parser.add_argument('--eval_ground_truth', action='store_true', default=True, help='Evaluate ground truth.')
        parser.add_argument('--code_completion', action='store_true', default=True, help='Perform code completion.')
        parser.add_argument('--eval_gen_code', action='store_true', default=True, help='Evaluate generated code.')
        parser.add_argument('--get_task_labels', action='store_true', default=True, help='Get task labels.')
        parser.add_argument("--first_run", action='store_true', default=False, help='Run the first time of a task to generate the dependency labels.')
        args = parser.parse_args()
        if args.first_run:
            args.eval_ground_truth = True
            args.code_completion = False
            args.eval_gen_code = False
            args.get_task_labels = True

        main(args, args.csv_folder_path, args.output_file_path, args.gen_model, 
                code_gen_mode = args.code_gen_mode, short_range = args.short_range, medium_range = args.medium_range, 
                eval_ground_truth=args.eval_ground_truth, code_completion=args.code_completion, eval_gen_code=args.eval_gen_code, get_task_labels=args.get_task_labels)
    # finally:
    #     # Kill the process at the end
    #     os.kill(pid, signal.SIGTERM)