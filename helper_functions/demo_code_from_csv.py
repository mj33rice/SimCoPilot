import os
import sys
import csv
import argparse
import subprocess
import ast
from termcolor import colored
from text_split_and_color import colored_line_numbers_and_text, add_line_numbers_and_color

def parse_frequency_data(frequency_analysis):
    try:
        return ast.literal_eval(frequency_analysis)
    except (ValueError, SyntaxError):
        return {}

def get_max_range_label(keys):
    range_order = ['Short-Range', 'Medium-Range', 'Long-Range']
    max_label = None
    for key in keys:
        for label in key.split():
            if label in range_order and (max_label is None or range_order.index(label) > range_order.index(max_label)):
                max_label = label
    return max_label

def label_present_in_analysis(label, analysis_data, label_category, use_max_range=True):
    analysis_dict = parse_frequency_data(analysis_data)
    # import pdb;pdb.set_trace()
    if label_category == 'reason':
        return any(label == key for key in analysis_dict)
    elif label_category == 'horizon':
        if use_max_range:
            max_label = get_max_range_label(analysis_dict.keys())
            return label == max_label
        else:
            return any(label in key.split() for key in analysis_dict)
    return False

def find_csv_files(base_dir, model_name, code_gen_mode, code_task, timestamp=None):
    # Construct the file prefix and directory path
    file_prefix = f"{model_name}_{code_gen_mode}_{code_task}"
    # Use base_dir as dir_path
    dir_path = base_dir
    # Check for timestamp to define the pattern
    if timestamp:
        file_pattern = f"{file_prefix}_{timestamp}.csv"
    else:
        file_pattern = f"{file_prefix}_"

    # List all files in the specified directory
    try:
        files = os.listdir(dir_path)
    except FileNotFoundError:
        print(f"Directory {dir_path} not found.")
        return []

    # Filter out files that match the pattern
    matching_files = [os.path.join(dir_path, f) for f in files if f.startswith(file_pattern) and f.endswith('.csv')]
    return matching_files

def print_analysis_results(row):
    if 'gen_code_pass_ratio' in row and row.get('gen_code_pass_ratio', '') != '':
        print(colored("Generated Code Pass Ratio:", "yellow"))
        # Evaluate gen_code_pass_ratio and print in green if it equals 1, otherwise print in red
        pass_ratio_eval = eval(row.get('gen_code_pass_ratio', '0')) == 1
        print(colored(row.get('gen_code_pass_ratio', 'N/A'), "green" if pass_ratio_eval else "red"))
    
    if 'gen_code_eval_res' in row and row.get('gen_code_eval_res', '') != '':
        print(colored("Generated Code Evaluation Results:", "yellow"))    
        eval_res = eval(row.get('gen_code_eval_res', '[]')) if row.get('gen_code_eval_res', '').startswith('[') else row.get('gen_code_eval_res', '').split('\n')
        for result in eval_res:
            color = "green" if result.startswith("Success:") else "red"
            print(colored(result, color))

def print_csv_content(file_name, search_label=None, label_category=None, use_max_range=True):
    # Open and read the CSV file
    with open(file_name, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            analysis_col = 'reason_freq_analysis' if label_category == 'reason' else 'horizon_freq_analysis'
            if search_label and not label_present_in_analysis(search_label, row.get(analysis_col, '{}'), label_category, use_max_range):
                continue  # Skip this row if the label is not found in the specified analysis

            # Convert string representations of lists back into actual lists for before and after code
            before_code = eval(row['before']) if row['before'].startswith('[') else row['before'].split('\n')
            after_code = eval(row['after']) if row['after'].startswith('[') else row['after'].split('\n')
            between_code = eval(row['between']) if row['between'].startswith('[') else row['between'].split('\n')
            # import pdb;pdb.set_trace()
            # The line number for the first line of code in 'before_code'
            start_line = 1 + len(before_code)  # Adjust if your CSV includes specific line numbers
            end_line = start_line + len(between_code) - 1 # The line number for the last line of code in 'before_code'

            # Use helper functions to color code lines
            colored_before, colored_between, colored_after = colored_line_numbers_and_text(before_code, between_code, after_code, start_line, end_line)
            print(colored("##############  Ground Truth Before ##############", "yellow"))
            print(colored_before)
            # Print formatted code sections
            print(colored("##############  Ground Truth Between ##############", "yellow"))
            print(colored(colored_between, "cyan"))
            if 'gen_code_selected' in row and row['gen_code_selected']:
                gen_code = eval(row['gen_code_selected']) if row['gen_code_selected'].startswith('[') else row['gen_code_selected'].split('\n')
                colored_gen_code = add_line_numbers_and_color(gen_code, start=start_line, text_color="red")  # start from the end of before_code
                print(colored("############### Generated Code ###############", "yellow"))
                print(colored(colored_gen_code, "red"))
            print(colored("##############  Ground Truth After ##############", "yellow"))
            print(colored(colored_after, "blue"))

            print(colored("############## Analysis Results ##############", "yellow"))
            # import pdb;pdb.set_trace()
            if 'reason_freq_analysis' in row:
                print(colored("Reason Categories:", "yellow"))
                print(row.get('reason_categories_output', 'N/A'))
            if 'horizon_freq_analysis' in row:
                print(colored("Horizon Categories:", "yellow"))
                print(row.get('horizon_categories_output', 'N/A'))

            print(colored("############## Task Categories ##############", "yellow"))
            # Assuming analysis results are stored as strings
            if 'reason_freq_analysis' in row:
                reason_freq_analysis = row['reason_freq_analysis']
                print(colored("Reason Categories:", "yellow"))
                print(colored(reason_freq_analysis, "blue"))
            if 'horizon_freq_analysis' in row:
                horizon_freq_analysis = row['horizon_freq_analysis']
                print(colored("Horizon Categories:", "yellow"))
                print(colored(horizon_freq_analysis, "blue"))

            print_analysis_results(row)  # Separate function for analysis results
            import pdb;pdb.set_trace()

def main(args):
    # Determine the category of the label
    label_category = None
    if args.label in ['List_Comprehension', 'Lambda_Expressions', 'Generator_Expressions', 'If-else Reasoning', 'Stream_Operations', 'Define Stop Criteria']:
        label_category = 'reason'
    elif args.label in ['Short-Range', 'Medium-Range', 'Long-Range', 'Variable', 'Global_Variable', 'Function', 'Class', 'Library']:
        label_category = 'horizon'

    matching_files = find_csv_files(args.base_dir, args.model_name, args.code_gen_mode, args.code_task, args.timestamp)
    # import pdb;pdb.set_trace()
    if matching_files:
        for file_name in matching_files:
            print(f"Displaying content from {file_name}:")
            print_csv_content(file_name, args.label, label_category, args.use_max_range)
            print("\n")  # Add a newline for better separation between files
    else:
        print("No matching CSV files found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and display code generation analysis results from CSV files.')
    # Get the root directory of the repository
    root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')    
    # Construct the default base directory
    default_base_dir = os.path.join(root_dir, 'Analysis_Results', 'Post_process', 'Python')
    parser.add_argument('--base_dir', type=str, default=default_base_dir, help='Base directory where the analysis results are stored. Default is the root directory of the repository.')
    parser.add_argument('--model_name', type=str, default='gpt-4-0125-preview', help='Name of the language model. Default is "gpt-4-0125-preview".')
    parser.add_argument('--code_gen_mode', type=str, default='no_afterlines', help='Code generation mode. Default is "no_afterlines".')
    parser.add_argument('--code_task', type=str, default='simplex_method_w_comments_V2', help='Description of the code task. Default is "simplex_method_w_comments_V2".')
    parser.add_argument('--timestamp', type=str, help='Specific timestamp for file matching. Optional.')
    parser.add_argument('--label', type=str, default='List_Comprehension', help='Specific label to search for in the analysis results. Default is "List_Comprehension".')
    parser.add_argument('--use_max_range', action='store_true', default=True, help='Use the maximum range label for horizon categories.')
    args = parser.parse_args()

    main(args)