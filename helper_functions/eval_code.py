import os
import ast
import re
import sys
import json
import subprocess
import tempfile
import shutil
from termcolor import colored, cprint

def execute_with_timeout(merged_code, test_case, source_code_path, timeout=5*60, run_test_gpu_id=None):
    try:
        # Set the CUDA_VISIBLE_DEVICES environment variable
        if run_test_gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(run_test_gpu_id)
        if source_code_path.endswith('.py'):
            return execute_python_code(merged_code, test_case, source_code_path, timeout=timeout)
        elif source_code_path.endswith('.java'):
            return execute_java_code(merged_code, test_case, source_code_path, timeout=timeout)
        else:
            raise ValueError("Unsupported file type. Please provide a Python (.py) or Java (.java) file.")
    except subprocess.TimeoutExpired:
        return "Timeout: Execution exceeded time limit."
    except Exception as e:
        return f"Unexpected error during execution: {e}"

def execute_python_code(merged_code, test_case, source_code_path, timeout=5*60):
    test_dir = os.path.abspath(os.path.dirname(source_code_path))
    with tempfile.TemporaryDirectory() as temp_workspace:
        temp_python_file = write_python_to_temp_file(merged_code, test_case, temp_workspace, source_code_path)
        symlink_supporting_files(test_dir, temp_workspace)
        exec_res = run_python_code(temp_python_file, timeout=timeout)
        return exec_res

def write_python_to_temp_file(merged_code, test_case, temp_workspace, source_code_path):
    temp_python_file = os.path.join(temp_workspace, os.path.basename(source_code_path))
    with open(temp_python_file, 'w') as file:
        file.write(merged_code)
        file.write('\n')
        file.write(test_case)
    return temp_python_file

def run_python_code(temp_python_file, timeout=5*60):
    exec_command = [sys.executable, temp_python_file]
    exec_res = subprocess.run(exec_command, capture_output=True, text=True, cwd=os.path.dirname(temp_python_file), timeout=timeout)
    return exec_res.stdout if exec_res.returncode == 0 else f"Error: {exec_res.stderr}"

def execute_java_code(merged_code, test_case, source_code_path, timeout=5*60):
    class_name, test_dir, junit_jar_path = prepare_java_environment(source_code_path)
    with tempfile.TemporaryDirectory() as temp_workspace:
        temp_java_file = write_java_to_temp_file(merged_code, temp_workspace, source_code_path)
        symlink_supporting_files(test_dir, temp_workspace)
        classpath = construct_classpath(junit_jar_path, test_dir, temp_workspace)
        compile_res = compile_java_code(classpath, temp_java_file, temp_workspace)
        if compile_res.returncode != 0:
            return f"Error: Compilation error: {compile_res.stderr}"
        return run_java_code(classpath, class_name, junit_jar_path, temp_workspace, test_case, timeout=timeout)

def prepare_java_environment(source_code_path):
    class_name = os.path.splitext(os.path.basename(source_code_path))[0]
    test_dir = os.path.abspath(os.path.dirname(source_code_path))
    junit_jar_path = ""
    if "COMP215" in source_code_path:
        comp215_dir = test_dir.split("COMP215")[0] + "COMP215"
        junit_jar_path = os.path.join(comp215_dir, "junit-platform-console-standalone-1.9.3.jar")
    return class_name, test_dir, junit_jar_path

def write_java_to_temp_file(merged_code, temp_workspace, source_code_path):
    temp_java_file = os.path.join(temp_workspace, os.path.basename(source_code_path))
    with open(temp_java_file, 'w') as file:
        file.write(merged_code)
    return temp_java_file

def symlink_supporting_files(test_dir, temp_workspace):
    for item in os.listdir(test_dir):
        source_item = os.path.join(test_dir, item)
        dest_item = os.path.join(temp_workspace, item)
        if not (item.endswith('.java') or item.endswith('.py')) and not os.path.exists(dest_item):
            os.symlink(source_item, dest_item)

def construct_classpath(junit_jar_path, test_dir, temp_workspace):
    extra_jars = ":".join([os.path.join(test_dir, jar) for jar in os.listdir(test_dir) if jar.endswith('.jar')])
    return ":".join(filter(None, [junit_jar_path, extra_jars, temp_workspace]))

def compile_java_code(classpath, temp_java_file, temp_workspace):
    return subprocess.run(["javac", "-cp", classpath, temp_java_file], capture_output=True, text=True, cwd=temp_workspace)

def run_java_code(classpath, class_name, junit_jar_path, temp_workspace, test_case, timeout=5*60):
    if junit_jar_path:
        exec_command = ["java", "-cp", classpath, "org.junit.platform.console.ConsoleLauncher", "--select-class", class_name]
    else:
        exec_command = ["java", "-cp", classpath, class_name] + test_case.split(" ")[2:]
    exec_res = subprocess.run(exec_command, capture_output=True, text=True, cwd=temp_workspace, timeout=timeout)
    # return exec_res.stdout if exec_res.returncode == 0 else f"Error: {exec_res.stderr}"
    if exec_res.returncode != 0:
        return f"Error: {exec_res.stderr}"
    else:
        return exec_res.stdout


def create_json_from_java(java_file_path):
    # Check if the given path is a file and has a .java extension
    if not os.path.isfile(java_file_path) or not java_file_path.endswith('.java'):
        print("The provided path does not lead to a Java file.")
        return
    # Extracting directory and filename without extension
    directory, filename = os.path.split(java_file_path)
    file_base_name = filename[:-5]  # Remove '.java' extension
    # Constructing the JSON file path
    json_file_path = os.path.join(directory, file_base_name + '.json')
    return json_file_path

def save_output_to_json(expected_output, file_path):
    # Define the dictionary structure as you described
    data = {
        "test_case1": "",
        "expected_output1": expected_output
    }

    # Writing the dictionary to a file in JSON format
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def compare_output(actual_output, expected_output):
    final_actual_output_dict = ast.literal_eval(actual_output)
    final_expected_output_dict = ast.literal_eval(expected_output)
    return final_actual_output_dict == final_expected_output_dict

def execute_and_compare(before, target, after, test_case, expected_output, source_code_path, timeout=5*60):
    """
    Merges the given code segments, executes them, and compares the output to the expected output.
    Returns both the comparison results for final output and all outputs.
    """
    merged_code = '\n'.join(before + target + after)
    actual_output = execute_with_timeout(merged_code, test_case, source_code_path, timeout=timeout)
    # save_output_to_json(actual_output, create_json_from_java(source_code_path))

    if actual_output.startswith("Error") or actual_output.startswith("Timeout"):
        return actual_output, actual_output  # Return the error for both comparisons
    # Final output comparison
    final_actual_output = next(filter(None, actual_output.strip().split('\n')[-1:]), '').strip()
    final_expected_output = next(filter(None, expected_output.strip().split('\n')[-1:]), '').strip()
    final_output_comparison = "Success: Final output matches the expected results." if final_actual_output == final_expected_output else "Failure: Final output does not match."
    # All outputs comparison
    normalized_actual = " ".join(actual_output.strip().split())
    normalized_expected = " ".join(expected_output.strip().split())
    all_outputs_comparison = "Success: All Output matches the expected results." if normalized_actual == normalized_expected else "Failure: All Output does not match."
    return (all_outputs_comparison, final_output_comparison)

    
def exe_eval(before, between, after, source_code_path, expected_result=None, test_case=None, convert_expected_result=True, eval_timeout=5*60, output_to_check='final_output'):
    if convert_expected_result:
        expected_result = expected_result.encode().decode('unicode_escape')
    (all_outputs_comparison, final_output_comparison) = execute_and_compare(before, between, after, test_case, expected_result, source_code_path, timeout=eval_timeout)
    print(colored("############## Code exe eval ##############", "yellow"))

    if output_to_check == 'final_output':
        print(colored(final_output_comparison,"green"))
        print(colored("############## Code exe eval ##############", "yellow"))
        return final_output_comparison
    elif output_to_check == 'all_output':
        print(colored(all_outputs_comparison,"green"))
        print(colored("############## Code exe eval ##############", "yellow"))
        return all_outputs_comparison

def load_test_cases_w_expected_output(file_path, before, between, after, source_code_path, output_to_check, print_process=False, eval_timeout=5*60):
    """
    Loads the test cases and expected outputs from the given JSON file.
    """
    final_eval_result = []
    # Read the file
    with open(file_path, 'r') as file:
        data = file.read()

    # Parse the JSON
    test_dict = json.loads(data)

    # Initialize an index for looping through test cases
    index = 1

    # Loop through all the test_cases and expected_outputs
    while True:
        # Construct the keys for the current test case and expected output
        test_case_key = f"test_case{index}"
        expected_output_key = f"expected_output{index}"
        
        # Check if both the test case and expected output exist in the dictionary
        if test_case_key in test_dict and expected_output_key in test_dict:
            # Get the test case and expected output from the dictionary
            test_case = test_dict[test_case_key]
            expected_output = test_dict[expected_output_key]
            if print_process:
                # (Optional) Use the test case and expected output here, e.g., print or further processing
                print(f"Running {test_case_key}:")
                print(test_case)
                print(f"Expected Output for {test_case_key}:")
                print(expected_output)
                print("\n")  # Just for better readability

            eval_result = exe_eval(before, between, after, source_code_path, expected_result=expected_output, test_case=test_case, convert_expected_result=True, eval_timeout=eval_timeout, output_to_check=output_to_check)

            final_eval_result.append(eval_result)
            
            # Move to the next index
            index += 1
        else:
            # if index == 1 means none of the test cases or expected outputs were found
            if index == 1:
                raise ValueError(f"Test case or expected output not found for index {index}.")
            # If the test case or expected output does not exist, break the loop
            break
    pass_ratio = calculate_pass_ratio(final_eval_result)
    
    return final_eval_result, pass_ratio

def calculate_pass_ratio(test_outputs):
    # Count the number of successes and total tests
    success_count = sum(1 for output in test_outputs if output.startswith('Success'))
    total_tests = len(test_outputs)

    # Create a ratio string
    pass_ratio_str = f"{success_count}/{total_tests}"

    return pass_ratio_str

def compile_and_run_tests(base_dir, assignments):
    results = {}
    for assignment, details in assignments.items():
        # Extracting the path for junit-platform-console-standalone JAR
        junit_jar_path = os.path.join(base_dir, "COMP215", "junit-platform-console-standalone-1.9.3.jar")
        
        # Constructing the test directory path
        test_dir = os.path.join(base_dir, assignment, "Test")
        java_file = details['java_file']

        # Building the classpath with the junit jar and any specified extra jars
        extra_jars = ":".join([os.path.join(test_dir, jar) for jar in details.get('extra_jars', [])])
        classpath = f"{junit_jar_path}"
        if extra_jars:
            classpath += f":{extra_jars}"

        # Compile the Java file with the specified classpath
        compile_cmd = ["javac", "-cp", classpath, os.path.join(test_dir, java_file)]
        subprocess.run(compile_cmd, check=True, cwd=test_dir)

        # Running the tests with the specified classpath
        run_classpath = f"{junit_jar_path}:{test_dir}"
        if extra_jars:
            run_classpath += f":{extra_jars}"
        run_cmd = ["java", "-jar", junit_jar_path, "-cp", run_classpath, "--select-class", java_file.replace(".java", "")]
        
        # Capturing the output of the test run
        run_result = subprocess.run(run_cmd, capture_output=True, text=True, cwd=test_dir)

        # Parsing the output to extract test results
        test_info = parse_COMP215_test_output(run_result.stdout)
        pass_ratio = calculate_COMP215_pass_ratio(test_info)

        results[assignment] = pass_ratio

    return results

def calculate_COMP215_pass_ratio(test_info):
    """Calculate the pass ratio based on the parsed test information."""
    if test_info['tests_found'] > 0:
        return test_info['tests_successful'] / test_info['tests_found']
    else:
        return 0
    
def parse_COMP215_test_output(output):
    """Parse the test run output to extract the number of successful and total tests."""
    test_info = {
        'tests_successful': 0,
        'tests_found': 0
    }
    # Using regular expressions to find the relevant information
    successful_match = re.search(r"\[\s*(\d+) tests successful\s*\]", output)
    found_match = re.search(r"\[\s*(\d+) tests found\s*\]", output)

    if successful_match:
        test_info['tests_successful'] = int(successful_match.group(1))
    if found_match:
        test_info['tests_found'] = int(found_match.group(1))

    return test_info
############################################################################################################
# # Define assignments with their respective Java files and any extra JAR dependencies
# assignments = {
#     # "COMP215/A0": {"java_file": "FactorizationTester.java"},
#     # "COMP215/A1": {"java_file": "CounterTester.java"},
#     "COMP215/A2": {"java_file": "DoubleVectorTester.java", "extra_jars": ["sparsearray.jar"]},
#     # "COMP215/A3": {"java_file": "SparseArrayTester.java", "extra_jars": ["sparsearray.jar"]},
#     # "COMP215/A4": {"java_file": "DoubleMatrixTester.java", "extra_jars": ["doublevector.jar"]},
#     # "COMP215/A5": {"java_file": "RNGTester.java", "extra_jars": ["a5.jar"]},
#     # "COMP215/A6": {"java_file": "TopKTester.java"},
#     # "COMP215/A7": {"java_file": "MTreeTester.java", "extra_jars": ["AVLTopKMachine.jar", "doublevector.jar", "mtree.jar"]},
# }


# BASE_DIR = "/Users/charles/Documents/Work_w_Chris/Transformer/GPT/Eval_Prog/example_code/Java"
# # Running the tests and getting the pass ratios
# test_results = compile_and_run_tests(BASE_DIR, assignments)

# # Printing the pass ratios for each assignment
# for assignment, pass_ratio in test_results.items():
#     print(f"{assignment}: Pass Ratio = {pass_ratio:.2f}")
############################################################################################################

# # Example usage
# success_outputs = ['Success: Output matches the expected results.', 'Success: Output matches the expected results.', 'Success: Output matches the expected results.']
# error_outputs = ['Error:   File "<string>", line 73\n    print("Initial tableau:")\n                             ^\nIndentationError: unindent does not match any outer indentation level\n', 'Error:   File "<string>", line 73\n    print("Initial tableau:")\n                             ^\nIndentationError: unindent does not match any outer indentation level\n', 'Error:   File "<string>", line 73\n    print("Initial tableau:")\n                             ^\nIndentationError: unindent does not match any outer indentation level\n']

# # Calculate and print pass ratios
# print("Pass ratio for success_outputs:", calculate_pass_ratio(success_outputs))  # Expected: 3/3
# print("Pass ratio for error_outputs:", calculate_pass_ratio(error_outputs))      # Expected: 0/3

# load_test_cases_w_expected_output('../example_code/Simplex_Method/test_cases.json')

# #Example usage with your provided before, target, and after parts
# before_part = ['import heapq', '', 'def dot(a,b):', '   return sum(x*y for x,y in zip(a,b))', '', 'def column(A, j):', '   return [row[j] for row in A]', '', 'def transpose(A):', '   return [column(A, j) for j in range(len(A[0]))]', '', 'def isPivotCol(col):', '   return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1', '', 'def variableValueForPivotColumn(tableau, column):', '   pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]', '   return tableau[pivotRow][-1]', '', 'def canImprove(tableau):', '   lastRow = tableau[-1]', '   return any(x > 0 for x in lastRow[:-1])', '', '# this can be slightly faster', 'def moreThanOneMin(L):', '   if len(L) <= 1:', '      return False', '   x,y = heapq.nsmallest(2, L, key=lambda x: x[1])', '   return x == y', '', 'def identity(numRows, numCols, val=1, rowStart=0):', '   return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] ', '', '', 'def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):', '   newVars = 0', '   numRows = 0', '   if gtThreshold != []:', '      newVars += len(gtThreshold)', '      numRows += len(gtThreshold)', '   if ltThreshold != []:']
# target_part = ['      newVars += len(ltThreshold)', '      numRows += len(ltThreshold)', '   if eqThreshold != []:', '      numRows += len(eqThreshold)', '   if not maximization:', '      cost = [-x for x in cost]', '   if newVars == 0:', '      return cost, equalities, eqThreshold', '']
# after_part = ['   newCost = list(cost) + [0] * newVars', '   constraints = []', '   threshold = []', '   oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]', '   offset = 0', '', '   for constraintList, oldThreshold, coefficient in oldConstraints:', '      constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]', '      threshold += oldThreshold', '      offset += len(oldThreshold)', '   return newCost, constraints, threshold', '', "'''", '   simplex: [float], [[float]], [float] -> [float], float', '   Solve the given standard-form linear program:', '      max <c,x>', '      s.t. Ax = b', '           x >= 0', '   providing the optimal solution x* and the value of the objective function', "'''", 'def simplex(c, A, b):', '   # assume the last m columns of A are the slack variables; the initial basis is the set of slack variables', '   tableau = [row[:] + [x] for row, x in zip(A, b)]', '   tableau.append([ci for ci in c] + [0])', '   print("Initial tableau:")', '   for row in tableau:', '      print(row)', '   print()', '', '   while canImprove(tableau):', '      # pick minimum positive index of the last row', '      column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]', '      column = min(column_choices, key=lambda a: a[1])[0]', '', '      # check if unbounded', '      if all(row[column] <= 0 for row in tableau):', "         raise Exception('Linear program is unbounded.')", '', '      # check for degeneracy: more than one minimizer of the quotient', '      quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]', '', '      if moreThanOneMin(quotients):', "         raise Exception('Linear program is degenerate.')", '', '      # pick row index minimizing the quotient', '      row = min(quotients, key=lambda x: x[1])[0]', '', '      pivot = row, column', '', '      print("Next pivot index is=%d,%d \\n" % pivot)', '      i,j = pivot', '      pivotDenom = tableau[i][j]', '      tableau[i] = [x / pivotDenom for x in tableau[i]]', '', '      for k,row in enumerate(tableau):', '         if k != i:', '            pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]', '            tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]', '      print("Tableau after pivot:")', '      for row in tableau:', '         print(row)', '      print()', '   ', '   # the pivot columns denote which variables are used', '   columns = transpose(tableau)', '   indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]', '   primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]', '   objective_value = -(tableau[-1][-1])', '   return tableau, primal_solution, objective_value', '', 'if __name__ == "__main__":', '   c = [12, 16]', '   A = [[10, 20], [8, 8]]', '   b = [120, 80]', '', '   # add slack variables by hand', '   A[0] += [1,0]', '   A[1] += [0,1]', '   c += [0,0]', '', '   t, s, v = simplex(c, A, b)', '   print(s)', '   print(v)']
# expected_result = "your_expected_result"
    
# # # Example usage
# # before_part = ["print('Hello')", "x = 5"]
# # target_part = ["print('Calculating...')", "y = x + 2"]
# # after_part = ["print('Result:', y * 2)"]
# # expected_result = "Hello\nCalculating...\nResult: 14\n"
# result = execute_and_compare(before_part, target_part, after_part, expected_result)
# print(result)

