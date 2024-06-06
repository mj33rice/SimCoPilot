import ast
import json
import csv
import os
import datetime
import argparse
import traceback  # Import traceback to log the exceptions if needed
from dependency_analyzer.analyzer import DependencyAnalyzer
from dependency_analyzer.Java_analyzer import JavaDependencyAnalyzer
from termcolor import colored, cprint
from helper_functions.text_split_and_color import split_code, colored_line_numbers_and_text, add_line_numbers_and_color
from helper_functions.eval_code import execute_and_compare, load_test_cases_w_expected_output
from helper_functions.LLMs_gen import LLMs_gen_and_post_process, extract_code_results
from helper_functions.dependency_result_analysis import analyze_and_get_frequencies, define_patterns
import signal

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use the first GPU
# Get the current process id
pid = os.getpid()

def modify_output_path(source_code_path, output_file_path, gen_model, code_gen_mode):
    # Extract the last part of the file name from source_code_path and remove the extension
    file_name_without_extension = os.path.splitext(os.path.basename(source_code_path))[0]

    # Get current time in the format "M_D_H_M"
    current_time = datetime.datetime.now().strftime("%m_%d_%H_%M")

    # Construct the new file name with gen_model, code_gen_mode, the last part of the file name without extension, and current time
    new_file_name = f"{gen_model}_{code_gen_mode}_{file_name_without_extension}_{current_time}.csv"

    # Construct the new output path by joining the new file name with the directory of the original output path
    output_dir = os.path.dirname(output_file_path)
    new_output_path = os.path.join(output_dir, new_file_name)

    return new_output_path, file_name_without_extension

def get_checkppoints_LNs(program_type, task_list, check_points_type='checkpoint_LN_infilling'):
    # Load the data from the JSON file
    file_path = f"example_code/{program_type}/{program_type}_tasks_checkpoints.json"
    with open(file_path, 'r') as f:
        tasks = json.load(f)

    # Extract the check_points_type values for the tasks in task_list
    interesting_points_L = []
    output_to_check = None
    for task in tasks:
        if task['Task_Name'] in task_list:
            interesting_points_L.extend(task[check_points_type])
            if 'output_to_check' in task:
                output_to_check = task['output_to_check']

    return interesting_points_L, output_to_check

def get_checkpoints_labels(args, program_type, source_code, start_line, end_line, short_range, medium_range, print_results=True):
    print(colored("############## Analysis Results ##############", "yellow"))
    # Initialize the analyzer with the source code and the start and end lines for the analysis
    # Determine the file extension
    if program_type == "Python":
        analyzer = DependencyAnalyzer(source_code, start_line, end_line, short_range, medium_range)
        # Perform the dependency analysis
        reason_categories_output, horizon_categories_output, reason_categories_printout, horizon_categories_printout = analyzer.analyze_dependencies(args.source_code_path, read_from_results=args.read_dependency_results, save_results=args.save_dependency_results)

    elif program_type == "Java":
        analyzer = JavaDependencyAnalyzer(source_code, start_line, end_line, short_range, medium_range, update_def_line=args.update_def_line)
        # Perform the dependency analysis
        reason_categories_output, horizon_categories_output, reason_categories_printout, horizon_categories_printout = analyzer.analyze_dependencies(args.source_code_path, read_from_results=args.read_dependency_results, save_results=args.save_dependency_results)
    
    else:
        raise ValueError("Unsupported file type. Please provide a Python (.py) or Java (.java) file.")

    horizon_pattern, reason_pattern = define_patterns()
    horizon_freq_analysis = analyze_and_get_frequencies(horizon_categories_printout, None, horizon_pattern)
    reason_freq_analysis = analyze_and_get_frequencies(reason_categories_printout, None, reason_pattern)
    if print_results:
        print(colored("############## Task Categories ##############", "yellow"))
        print(colored('Reason Categories: ',"yellow"))
        print(colored(reason_freq_analysis,"green"))
        print(colored('Horizon Categories: ',"yellow"))
        print(colored(horizon_freq_analysis,"green"))
    return reason_categories_output, horizon_categories_output,\
            reason_categories_printout, horizon_categories_printout,\
            horizon_freq_analysis, reason_freq_analysis

def analyze_code(args, item, source_code, eval_ground_truth, test_cases_path, source_code_path, 
                 program_type, output_to_check, code_completion, eval_gen_code, get_task_labels,
                 gen_model, code_gen_mode, short_range, medium_range):
    start_line = item[0]
    end_line = item[1]
    before, between, after = split_code(source_code, start_line, end_line)
    ground_truth_eval_res, ground_truth_pass_ratio = None, None
    if eval_ground_truth:
        #Eval code with ground truth:
        ground_truth_eval_res, ground_truth_pass_ratio = load_test_cases_w_expected_output(test_cases_path, before, between, after, source_code_path, output_to_check, eval_timeout=args.eval_timeout)

    colored_before, colored_between, colored_after = colored_line_numbers_and_text(before, between, after, start_line, end_line)
    print(colored_before)
    gen_code_dict, post_process_steps, selected_last_post_process_step, gen_code_selected, colored_gen_code = None, None, None, None, None
    if code_completion:
        #LLMs generation and post processing steps
        gen_code_dict, post_process_steps = LLMs_gen_and_post_process(before, after, program_type, 
                                            clean_gen_code = True, remove_gen_repeated_lines=True, add_auto_indent=True, print_post_process=True, 
                                            gen_mode=code_gen_mode, model=gen_model, 
                                            gen_time_out = args.code_gen_timeout)
        # Here you can specify steps explicitly, e.g., ['cleaned_code', 'trimmed_code']
        # Or leave it None to automatically pick the last non-empty result
        selected_last_post_process_step, gen_code_selected = extract_code_results(gen_code_dict, last_post_process_step = ['cleaned_code']) # You can also pass specific steps as a list
        gen_code_selected = gen_code_selected.split('\n')
        colored_gen_code = add_line_numbers_and_color(gen_code_selected, start=start_line, text_color="red")
    print(colored("##############  Ground Truth  ##############", "yellow"))
    print(colored(colored_between,"cyan"))
    if code_completion:
        print(colored("############## Generated Code ##############", "yellow"))
        print(colored(colored_gen_code,"red"))
    print(colored("##############  Ground Truth  ##############", "yellow"))
    print(colored_after)

    gen_code_eval_res, gen_code_pass_ratio = None, None
    if eval_gen_code:
        #Eval code with test cases:
        gen_code_eval_res, gen_code_pass_ratio  = load_test_cases_w_expected_output(test_cases_path, before, gen_code_selected, after, source_code_path, output_to_check, eval_timeout=args.eval_timeout)
    
    reason_categories_output, horizon_categories_output, reason_categories_printout, horizon_categories_printout, horizon_freq_analysis, reason_freq_analysis = None, None, None, None, None, None
    if get_task_labels:
        reason_categories_output, horizon_categories_output,\
        reason_categories_printout, horizon_categories_printout,\
        horizon_freq_analysis, reason_freq_analysis = get_checkpoints_labels(args, program_type, source_code, start_line, end_line, short_range, medium_range)
    return  start_line, end_line, before, between, after,\
            ground_truth_eval_res, ground_truth_pass_ratio,\
            gen_code_dict, post_process_steps, selected_last_post_process_step, gen_code_selected,\
            gen_code_eval_res, gen_code_pass_ratio,\
            reason_categories_output, horizon_categories_output,\
            reason_categories_printout, horizon_categories_printout,\
            horizon_freq_analysis, reason_freq_analysis

def main(args, source_code_path, test_cases_path, output_file_path, gen_model, code_gen_mode='no_afterlines', short_range=5, medium_range=20, eval_ground_truth=False, code_completion=True, eval_gen_code=True, get_task_labels=True):
    # Read the source code from file
    with open(source_code_path, 'r') as file:
        source_code = file.read()
    reason_freq_analysis = None
    horizon_freq_analysis = None
    reason_categories_printout = None
    horizon_categories_printout = None
    # Determine the file extension
    if source_code_path.endswith('.py'):
        program_type = "Python"
    elif source_code_path.endswith('.java'):
        program_type = "Java"
    else:
        raise ValueError("Unsupported file type. Please provide a Python (.py) or Java (.java) file.")

    new_output_path, file_name_without_extension = modify_output_path(source_code_path, output_file_path, gen_model, code_gen_mode)
    # Load the data from the JSON file
    check_points_type = "LN_range" if args.first_run else "checkpoint_LN_Completion" if code_gen_mode == 'no_afterlines' else "checkpoint_LN_infilling" if code_gen_mode == 'with_afterlines' else None
    if check_points_type is not None:
        interesting_points_L, output_to_check = get_checkppoints_LNs(program_type, [file_name_without_extension], check_points_type=check_points_type)
    else:
        raise ValueError("Invalid code_gen_mode. Please choose either 'no_afterlines' or 'with_afterlines'.")
    
    # import pdb;pdb.set_trace()
    if args.show_task_specs:
        print("##############  Output Path  ##############")
        print(new_output_path)
        print("##############  Code Task  ##############")
        print(file_name_without_extension)
        print('##############  Interesting Points  ##############')
        print(interesting_points_L)
        print("##############  Gen Model  ##############")
        print(gen_model)
        print("##############  Code Gen Mode  ##############")
        print(code_gen_mode)
        print("##############  Source Code Path  ##############")
        print(source_code_path)
        print("##############  Test Cases Path  ##############")
        print(test_cases_path)
        print("##############  Eval Ground Truth  ##############")
        print(eval_ground_truth)
        print("##############  Code Completion  ##############")
        print(code_completion)
        print("##############  Eval Gen Code  ##############")
        print(eval_gen_code)
        print("##############  Get Task Labels  ##############")
        print(get_task_labels)
        print("##############  Short Range  ##############")
        print(short_range)
        print("##############  Medium Range  ##############")
        print(medium_range)
        print("##############  Args  ##############")
        print(args)
        print('##############  Task Starts Here  ##############')
    if not args.skip_save_csv:
        # Open a CSV file to write the analysis results
        with open(new_output_path, mode='w', newline='') as file:
            # Define the column names
            fieldnames = [
                'start_line',
                'end_line',
                'code_task',
                'before',
                'between',
                'after',
                'gen_code_dict',
                'gen_code_process_steps',
                'selected_last_post_process_step',
                'gen_code_selected',
                'gen_code_eval_res',
                'gen_code_pass_ratio',
                'ground_truth_eval_res',
                'ground_truth_pass_ratio',
                'reason_categories_output',
                'horizon_categories_output',
                'reason_freq_analysis',
                'horizon_freq_analysis'
            ]
            
            # Create a writer object
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for item in interesting_points_L:
                # import pdb;pdb.set_trace()
                try:
                    start_line, end_line, before, between, after,\
                    ground_truth_eval_res, ground_truth_pass_ratio,\
                    gen_code_dict, post_process_steps, selected_last_post_process_step, gen_code_selected,\
                    gen_code_eval_res, gen_code_pass_ratio,\
                    reason_categories_output, horizon_categories_output,\
                    reason_categories_printout, horizon_categories_printout,\
                    horizon_freq_analysis, reason_freq_analysis = analyze_code(args, item, source_code, eval_ground_truth, test_cases_path, source_code_path, 
                                                                                program_type, output_to_check, code_completion, eval_gen_code, get_task_labels,
                                                                                gen_model, code_gen_mode, short_range, medium_range)

                    # Write a row to the CSV file
                    writer.writerow({
                        'start_line': start_line,
                        'end_line': end_line,
                        'code_task': file_name_without_extension,
                        'before': before,
                        'between': between,
                        'after': after,
                        'gen_code_dict': gen_code_dict if code_completion else "",
                        'gen_code_process_steps': post_process_steps if code_completion else "",
                        'selected_last_post_process_step': selected_last_post_process_step if code_completion else "",
                        'gen_code_selected': "\n".join(gen_code_selected) if code_completion else "",
                        'gen_code_eval_res': gen_code_eval_res if eval_gen_code else "",
                        'gen_code_pass_ratio': gen_code_pass_ratio if eval_gen_code else "",
                        'ground_truth_eval_res': ground_truth_eval_res if eval_ground_truth else "",
                        'ground_truth_pass_ratio': ground_truth_pass_ratio if eval_ground_truth else "",
                        'reason_categories_output': "\n".join(reason_categories_printout) if reason_categories_printout else "",
                        'horizon_categories_output': "\n".join(horizon_categories_printout)if horizon_categories_printout else "",
                        'reason_freq_analysis': reason_freq_analysis if reason_freq_analysis else "",
                        'horizon_freq_analysis': horizon_freq_analysis if horizon_freq_analysis else ""
                    })
                except Exception as e:
                    print(f"Error processing item {item}: {e}")
                    traceback.print_exc()  # This prints the stack trace to the console
    else:
        args.code_gen_timeout = 20*60
        args.eval_timeout = 10*60
        for item in interesting_points_L:
            # import pdb;pdb.set_trace()
            start_line, end_line, before, between, after,\
            ground_truth_eval_res, ground_truth_pass_ratio,\
            gen_code_dict, post_process_steps, selected_last_post_process_step, gen_code_selected,\
            gen_code_eval_res, gen_code_pass_ratio,\
            reason_categories_output, horizon_categories_output,\
            reason_categories_printout, horizon_categories_printout,\
            horizon_freq_analysis, reason_freq_analysis = analyze_code(args, item, source_code, eval_ground_truth, test_cases_path, source_code_path, 
                                                                        program_type, output_to_check, code_completion, eval_gen_code, get_task_labels,
                                                                        gen_model, code_gen_mode, short_range, medium_range)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Analyze code dependencies.')
        parser.add_argument('source_code_path', type=str, help='Path to the source code file.')
        parser.add_argument('test_cases_path', type=str, help='Path to the test cases file.')
        parser.add_argument('--output_file_path', type=str, default='./Analysis_Results/complete_output.csv', help='Analysis Results of Code Generation.')
        parser.add_argument('--short_range', type=int, default=10, help='Distance for Short-Range dependencies.')
        #medium_range was 20
        parser.add_argument('--medium_range', type=int, default=30, help='Distance for Medium-Range dependencies.')
        parser.add_argument('--gen_model', type=str, default='gpt-3.5-turbo-0125', help='LLM Selection for code generation task.')
        # model= 'gpt-4-1106-preview', "gpt-4-0125-preview", #gpt-4-1106-preview #gpt-4 #gpt-4-32k #gpt-4-0613 #gpt-3.5-turbo-0613 #gpt-4-0125-preview
        parser.add_argument('--code_gen_mode', type=str, default='with_afterlines', help='LLM mode for code generation task, such as no_afterlines, with_afterlines, no_instruction.')
        parser.add_argument('--eval_timeout', type=int, default=5*60, help='Timeout for evaluating the generated code.')
        parser.add_argument('--code_gen_timeout', type=int, default=3*60, help='Timeout for generating the code.')
        parser.add_argument('--skip_save_csv', action='store_true', default=False, help='Skip writing to the CSV file')

        parser.add_argument('--read_dependency_results', action='store_true', default=False, help='Read dependency results from a CSV file.')
        parser.add_argument('--save_dependency_results', action='store_true', default=False, help='Save dependency results to a CSV file.')
        parser.add_argument('--update_def_line', action='store_true', default=True,  help='Update the definition line of variables when value reassign.')
        parser.add_argument('--show_task_specs', action='store_true', default=True, help='Show the task specs in the output.')
        
        parser.add_argument('--eval_ground_truth', action='store_true', default=False, help='Evaluate ground truth.')
        parser.add_argument('--code_completion', action='store_true', default=True, help='Perform code completion.')
        parser.add_argument('--eval_gen_code', action='store_true', default=False, help='Evaluate generated code.')
        parser.add_argument('--get_task_labels', action='store_true', default=True, help='Get task labels.')
        parser.add_argument("--first_run", action='store_true', default=False, help='Run the first time of a task to generate the dependency labels.')
        args = parser.parse_args()
        if args.first_run:
            args.eval_ground_truth = True
            args.code_completion = False
            args.eval_gen_code = False
            args.get_task_labels = True

        main(args, args.source_code_path, args.test_cases_path, args.output_file_path, args.gen_model, 
                code_gen_mode = args.code_gen_mode, short_range = args.short_range, medium_range = args.medium_range, 
                eval_ground_truth=args.eval_ground_truth, code_completion=args.code_completion, eval_gen_code=args.eval_gen_code, get_task_labels=args.get_task_labels)
    finally:
        # Kill the process at the end
        os.kill(pid, signal.SIGTERM)