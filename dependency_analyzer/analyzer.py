import os
import pandas as pd
import warnings
import ast
import json
import argparse
import builtins

class DependencyAnalyzer(ast.NodeVisitor):
    def __init__(self, source_code, start_line, end_line, short_range, medium_range):
        self.source_code = source_code
        self.tree = ast.parse(source_code)
        self.start_line = start_line
        self.end_line = end_line
        self.short_range = short_range
        self.medium_range = medium_range
        self.current_listcomp_line = None
        self.current_genexp_line = None  # Track the current generator expression line
        self.scope_stack = [{'type': 'global', 'locals': {}, 'globals': {}, 'loop': {}, 'list_comp': {}, 'generator_exp': {}, 'lambda': {}, 'functions': {}, 'classes': {}, 'libraries': {}}]
        self.global_vars = set()
        self.usages = set()
        self.horizon_categories = []
        self.reason_categories = []
        self.previous_if_conditions = []
        # self.pattern_based_completions = set()
        self.debug_print = False
        self.filter_none_def_line = True
        self.csv_save_path = './Analysis_Results/Dependency_Records/Python/'

    def current_scope(self):
        return self.scope_stack[-1]

    def enter_scope(self, scope_type='general', name=None):
        # Inherit globals and initialize other dictionaries correctly
        new_scope = {
            'type': scope_type, 
            'name': name, 
            'locals': {}, 
            'globals': self.scope_stack[0]['globals'].copy(), 
            'loop': {}, 
            'list_comp': {},
            'generator_exp': {},
            'lambda': {}, 
            'functions': {}, 
            'classes': {}, 
            'libraries': self.scope_stack[0]['libraries'].copy()
        }
        if self.debug_print:
            print(f"Entering scope: {new_scope}")
        self.scope_stack.append(new_scope)

    def exit_scope(self):
        self.scope_stack.pop()
    
    def add_variable(self, name, line, scope_type):
        current_scope = self.current_scope()
        if self.debug_print:
            print(f"Adding {scope_type} '{name}' at line {line} to scope {current_scope['type']}")
        current_scope[scope_type][name] = line
    
    def visit_For(self, node):
        self.enter_scope('loop')
        if isinstance(node.target, ast.Name):
            self.add_variable(node.target.id, node.lineno, 'loop')
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self.add_variable(elt.id, node.lineno, 'loop')
        self.generic_visit(node)
        if self.start_line <= node.lineno <= self.end_line:
            # Add stop criteria detection for 'for' loop
            self.reason_categories.append({'reason_category': 'Define Stop Criteria', 'usage_line': node.lineno})
        self.exit_scope()


    def visit_While(self, node):
        self.enter_scope('loop')
        self.generic_visit(node)
        if self.start_line <= node.lineno <= self.end_line:
            self.reason_categories.append({'reason_category': 'Define Stop Criteria', 'usage_line': node.lineno })
        self.exit_scope()
    
    def visit_Import(self, node):
        # Tracking imported libraries
        for alias in node.names:
            lib_name = alias.asname or alias.name
            self.scope_stack[0]['libraries'][lib_name] = node.lineno

    def visit_ImportFrom(self, node):
        # Tracking specific imports from libraries
        for alias in node.names:
            imported_name = alias.name
            alias_name = alias.asname or alias.name
            full_name = f"{node.module}.{imported_name}"
            self.scope_stack[0]['libraries'][alias_name] = node.lineno

    def visit_FunctionDef(self, node):
        # Track function definition in the nearest enclosing scope that can contain it
        self.current_scope()['functions'][node.name] = node.lineno
        self.enter_scope('function', node.name)
        for arg in node.args.args:
            self.add_variable(arg.arg, node.lineno, 'locals')
        self.generic_visit(node)
        self.exit_scope()

    def visit_ClassDef(self, node):
        # Similar to visit_FunctionDef, track class definitions
        self.current_scope()['classes'][node.name] = node.lineno
        self.enter_scope('class', node.name)
        self.generic_visit(node)
        self.exit_scope()

    def visit_comprehension(self, node):
        # Ensure variables in the target are registered in the current scope
        # This applies to both list comprehensions and generator expressions
        if isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self.add_variable(elt.id, self.current_listcomp_line or self.current_genexp_line, 'list_comp' if self.current_listcomp_line else 'generator_exp')
        elif isinstance(node.target, ast.Name):
            self.add_variable(node.target.id, self.current_listcomp_line or self.current_genexp_line, 'list_comp' if self.current_listcomp_line else 'generator_exp')
        super().generic_visit(node)  # Continue visiting the iter and ifs parts

    def visit_GeneratorExp(self, node):
        self.enter_scope('generator_exp')
        self.current_genexp_line = node.lineno  # Set the current generator expression line
        # Check if the generator expression is within the analysis range and report it
        if self.start_line <= node.lineno <= self.end_line:
            self.reason_categories.append({'reason_category': 'Generator_Expressions', 'usage_line': node.lineno})
        for generator in node.generators:
            self.visit(generator)
        # Visit the element being iterated over
        self.visit(node.elt)
        self.current_genexp_line = None  # Reset the current generator expression line after visiting
        self.exit_scope()

    def visit_ListComp(self, node):
        self.enter_scope('list_comp', None)
        # The line number for variables defined in list comprehensions should be the line number of the ListComp node itself
        self.current_listcomp_line = node.lineno
        # Check if the list comprehension is within the analysis range and report it
        if self.start_line <= node.lineno <= self.end_line:
            self.reason_categories.append({'reason_category': 'List_Comprehension', 'usage_line': node.lineno})
        for generator in node.generators:
            self.visit(generator)
        # Visit the element being iterated over
        self.visit(node.elt)
        self.current_listcomp_line = None
        self.exit_scope()
        
    def visit_Lambda(self, node):
        self.enter_scope('lambda', None)  # Enter a new scope for lambda, no name is needed
        # Handle arguments for the lambda function
        if hasattr(node, 'args') and hasattr(node.args, 'args'):
            for arg in node.args.args:
                self.add_variable(arg.arg, node.lineno, 'locals')
        # Check if the lambda expression is within the analysis range and report it
        if self.start_line <= node.lineno <= self.end_line:
            self.reason_categories.append({'reason_category': 'Lambda_Expressions', 'usage_line': node.lineno})
        self.generic_visit(node)  # Visit the body of the lambda
        self.exit_scope()
    
    def visit_Global(self, node):
        for name in node.names:
            self.global_vars.add(name)
            self.scope_stack[0]['globals'][name] = node.lineno

    def visit_Assign(self, node):
        # Handle tuple unpacking in assignments
        for target in node.targets:
            if isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.add_variable(elt.id, node.lineno, 'locals')
            elif isinstance(target, ast.Name):
                self.add_variable(target.id, node.lineno, 'locals')
        self.generic_visit(node)
    
    def visit_Name(self, node):
        var_name = node.id
        found = False
        undefined = False  # Flag to indicate if a var might be undefined

        # Check if the variable usage is within the analysis range
        if not (self.start_line <= node.lineno <= self.end_line):
            return

        for scope in reversed(self.scope_stack):
            # Add handling for lambda scope to ensure variables used within lambda are reported
            if scope['type'] == 'lambda' and var_name in scope['locals']:
                self.report_usage(var_name, node.lineno, 'lambda', scope['locals'][var_name])
                found = True
                break
            elif var_name in scope['locals']:
                self.report_usage(var_name, node.lineno, 'local', scope['locals'][var_name])
                found = True
                break
            elif var_name in scope['functions']:
                self.report_usage(var_name, node.lineno, 'function', scope['functions'][var_name])
                found = True
                break
            elif var_name in scope['classes']:
                self.report_usage(var_name, node.lineno, 'class', scope['classes'][var_name])
                found = True
                break
            elif var_name in scope['libraries']:
                self.report_usage(var_name, node.lineno, 'library', scope['libraries'][var_name])
                found = True
                break
            elif var_name in scope['globals']:
                self.report_usage(var_name, node.lineno, 'global', scope['globals'][var_name])
                found = True
                break
            elif var_name in scope['list_comp'] or var_name in scope['generator_exp']:
                scope_type = 'list_comp' if var_name in scope['list_comp'] else 'generator_exp'
                self.report_usage(var_name, node.lineno, scope_type, scope[scope_type][var_name])
                found = True
                break
            elif var_name in scope['loop']:
                self.report_usage(var_name, node.lineno, 'loop', scope['loop'][var_name])
                found = True
                break
            else:
                undefined = True  # Mark as potentially undefined but don't break; keep looking

        # Only report as undefined if not found in any scope
        if not found and undefined and var_name not in dir(builtins):
            self.report_usage(var_name, node.lineno, 'undefined', None)


    # The main, get_range_type, and analyze_dependencies methods remain mostly unchanged.
    def visit_If(self, node):
        # Handle 'If-else Reasoning' and 'Pattern-Based Completion'
        if self.start_line <= node.lineno <= self.end_line:
            self.handle_if_else_reasoning(node)
            self.handle_pattern_based_completion(node)
        self.generic_visit(node)

    #Reason analysis helper functions    
    ############################################
    def handle_if_else_reasoning(self, node):
        # Check if the if-else statement is within the analysis range
        if self.start_line <= node.lineno <= self.end_line:
            self.reason_categories.append({'reason_category': 'If-else Reasoning', 'usage_line': node.lineno})

        # Additionally, check for 'else' and 'elif' within the analysis range
        for body in node.orelse:
            if isinstance(body, ast.If):  # This handles 'elif'
                if self.start_line <= body.lineno <= self.end_line:
                    self.reason_categories.append({'reason_category': 'If-else Reasoning', 'usage_line': body.lineno})
            elif body:  # This handles 'else'
                # Find the first statement in the else block
                first_statement = body if isinstance(body, ast.stmt) else body[0]
                if self.start_line <= first_statement.lineno <= self.end_line:
                    self.reason_categories.append({'reason_category': 'If-else Reasoning', 'usage_line': first_statement.lineno})
    
    def handle_pattern_based_completion(self, node):
        condition_repr = self.get_condition_representation(node)
        if condition_repr:
            # Check current condition against previous conditions
            for prev_condition_repr in self.previous_if_conditions:
                if condition_repr == prev_condition_repr:
                    self.reason_categories.append({'reason_category': 'Pattern-Based Completion', 'usage_line': node.lineno})
                    break
            self.previous_if_conditions.append(condition_repr)

    def get_condition_representation(self, node):
        # Returns a simplified representation of the condition for pattern comparison
        if isinstance(node.test, ast.Compare):
            left = self.get_node_representation(node.test.left)
            right = [self.get_node_representation(comp) for comp in node.test.comparators]
            return (left, tuple(right))
        return None
    
    def get_node_representation(self, node):
        # Simplifies the node to a basic representation for comparison
        if isinstance(node, ast.Name):
            return 'var'  # Represents a variable
        if isinstance(node, ast.List):
            return 'list'  # Represents a list
        if isinstance(node, ast.Num):
            return 'num'  # Represents a number
        return str(node)
    def get_range_type(self, use_line, def_line):
        if def_line is None or use_line is None:
            return 'N/A'  # Not applicable if def_line is None
        range_distance = abs(use_line - def_line)
        if range_distance <= self.short_range:
            return 'Short-Range'
        elif range_distance <= self.medium_range:
            return 'Medium-Range'
        else:
            return 'Long-Range'

    def save_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    
    def read_from_csv(self, path, filter_usage_line=True):
        """ Read data from a CSV file and optionally filter by usage line. """
        if os.stat(path).st_size <= 1:
            # print(f'File is empty: {path}')
            return pd.DataFrame()
        else:
            df = pd.read_csv(path)
            if filter_usage_line:
                # Filter rows where 'usage_line' is within the specified range
                df = df[(df['usage_line'] >= self.start_line) & (df['usage_line'] <= self.end_line)]
            return df.to_dict(orient='records')
    
    def report_horizon_analysis_results(self):
        horizon_categories_output = []
        for result in self.horizon_categories:
            var_name = result['var_name']
            var_line = result['usage_line']
            scope_type = result['scope_type']
            def_line = result['def_line']
            range_type = result['range_type']

            # Skipping specific cases as in the original report_usage
            if var_line == def_line and scope_type not in ['list_comp', 'generator_exp', 'lambda']:
                continue
            if scope_type == 'builtin':
                continue
            # Constructing and printing messages according to the scope type
            if scope_type == 'function':
                horizon_categories_output.append(f"Function '{var_name}' used at line {var_line} is defined at line {def_line} and has a {range_type} dependency.")
            elif scope_type == 'class':
                horizon_categories_output.append(f"Class '{var_name}' used at line {var_line} is defined at line {def_line} and has a {range_type} dependency.")
            elif scope_type == 'library':
                horizon_categories_output.append(f"Library '{var_name}' used at line {var_line} is imported at line {def_line} and has a {range_type} dependency.")
            elif scope_type == 'list_comp':
                horizon_categories_output.append(f"Variable '{var_name}' used at line {var_line} is part of a List_Comprehension defined at line {def_line} and has a {range_type} dependency.")
            elif scope_type == 'generator_exp':  # Handle generator expressions specifically
                horizon_categories_output.append(f"Variable '{var_name}' used at line {var_line} is part of a Generator_Expressions defined at line {def_line} and has a {range_type} dependency.")
            elif scope_type == 'lambda':  
                horizon_categories_output.append(f"Variable '{var_name}' used at line {var_line} is part of a Lambda_Expressions defined at line {def_line} and has a {range_type} dependency.")
            elif scope_type == 'loop':
                horizon_categories_output.append(f"Variable '{var_name}' used at line {var_line} is part of a Loop defined at line {def_line} and has a {range_type} dependency.")
            elif scope_type == 'global':
                horizon_categories_output.append(f"Global_Variable '{var_name}' used at line {var_line} is defined at line {def_line} and has a {range_type} dependency.")
            elif scope_type == 'local':
                horizon_categories_output.append(f"Variable '{var_name}' used at line {var_line} is defined at line {def_line} and has a {range_type} dependency.")
            else:
                horizon_categories_output.append(f"Variable '{var_name}' used at line {var_line} is undefined and has a {range_type} dependency.")
        return horizon_categories_output
    
    def report_reason_analysis_results(self):
        # Store reason categories in a list
        reason_categories_output = []
        for result in self.reason_categories:
            reason = result['reason_category']
            lineno = int(result['usage_line'])
            reason_categories_output.append(f"'{reason}' detected at line {lineno}.")
        return reason_categories_output
    
    def print_list_elements(self, my_list):
        for element in my_list:
            print(element)

    def report_usage(self, var_name, var_line, scope_type, def_line):
        # Adjusted logic to report usage based on type and definition line
        # Skipping specific cases
        if var_line == def_line and scope_type not in ['list_comp', 'generator_exp', 'lambda']:
            return
        if scope_type == 'builtin':
            return
        # Conditionally add usage based on filter_none_def_line and whether def_line is not None
        if not self.filter_none_def_line or (self.filter_none_def_line and def_line is not None):
            self.horizon_categories.append({
                'var_name': var_name,
                'usage_line': var_line,
                'def_line': def_line,
                'scope_type': scope_type,
            })
    def post_analysis(self):
        # Sorting the list of dictionaries by 'usage_line'
        self.horizon_categories = sorted(self.horizon_categories, key=lambda x: x['usage_line'])
        # Remove duplicate dictionaries from the list
        self.horizon_categories = self.remove_duplicate_dict(self.horizon_categories)
        # Sorting the list of dictionaries by 'usage_line'
        self.reason_categories = sorted(self.reason_categories, key=lambda x: x['usage_line'])
        # Remove duplicate dictionaries from the list
        self.reason_categories = self.remove_duplicate_dict(self.reason_categories)
    
    def update_dependency_range(self, intermediate_dependency):
        # Update the range_type for each record
        updated_dependency_results = []
        for item in intermediate_dependency:
            use_line = int(item['usage_line'])
            def_line = int(item['def_line']) if item['def_line'] != None else None
            range_type = self.get_range_type(use_line, def_line if def_line else -1)
            item['range_type'] = range_type
            updated_dependency_results.append(item)
        return updated_dependency_results
    
    def remove_duplicate_dict(self, my_list):
        seen = set()
        no_duplicates = []
        for item in my_list:
            serialized_item = json.dumps(item, sort_keys=True)
            if serialized_item not in seen:
                seen.add(serialized_item)
                no_duplicates.append(item)
        return no_duplicates
    
    def analyze_dependencies(self, file_path, read_from_results=False, save_results=True):
        # Extract the base name of the Java file from the file path
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # Prepare file paths for CSV files
        final_dependency_csv_path = f"{self.csv_save_path}{base_name}_dependency_final.csv"
        final_reason_csv_path = f"{self.csv_save_path}{base_name}_reason_final.csv"
        verified_dependency_csv_path = f"{self.csv_save_path}Verified/{base_name}_dependency_verified.csv"
        verified_reason_csv_path = f"{self.csv_save_path}Verified/{base_name}_reason_verified.csv"

        if read_from_results:
            # Reload the intermediate results after manual correction
            # Use the base_name, self.start_line, and self.end_line to filter the results
            intermediate_dependency = self.read_from_csv(verified_dependency_csv_path)
            intermediate_reason = self.read_from_csv(verified_reason_csv_path)
            # Update range types 
            self.horizon_categories = self.update_dependency_range(intermediate_dependency)
            self.reason_categories = intermediate_reason
        else:
            self.visit(self.tree)
            if self.debug_print:
                # Analysis code here: populate usages, horizon_categories, reason_categories based on walked data
                print("Final Scope Stack:", self.scope_stack)# Print the final scope stack
            # Process horizon and reason categories
            self.post_analysis()
            # Update range types after manual correction
            self.horizon_categories = self.update_dependency_range(self.horizon_categories)
            if save_results:
                self.save_to_csv(self.horizon_categories,final_dependency_csv_path)
                self.save_to_csv(self.reason_categories,final_reason_csv_path)

        # Report the analysis results
        horizon_categories_printout = self.report_horizon_analysis_results()
        reason_categories_printout = self.report_reason_analysis_results()
        # Report reason categories
        print("Reason Categories:")
        self.print_list_elements(reason_categories_printout)
        print()
        print("Horizon Categories:")
        self.print_list_elements(horizon_categories_printout)
        return self.reason_categories, self.horizon_categories, reason_categories_printout, horizon_categories_printout 

def main(file_path, short_range, medium_range):
    # Read the source code from file
    with open(file_path, 'r') as file:
        source_code = file.read()
    # Define the markers
    start_marker = "######### Analyze from here #########\n"
    end_marker = "######### Analyze End here #########\n"

    # Split the source code into lines
    source_lines = source_code.splitlines(keepends=True)

    # Find the start and end line numbers for analysis
    start_line = end_line = None
    for i, line in enumerate(source_lines):
        if start_marker.strip() in line.strip():
            start_line = i + 1
        elif end_marker.strip() in line.strip():
            end_line = i
            break

    # Adjust the start and end lines if markers are not found
    if start_line is None:
        warnings.warn("Start marker not found. Analyzing from the beginning of the file.")
        start_line = 0  # Start from the first line
    if end_line is None:
        warnings.warn("End marker not found. Analyzing till the end of the file.")
        end_line = len(source_lines)  # End at the last line
        
    if start_line is None or end_line is None:
        raise ValueError("Analysis markers not found in the source code.")

    # Parse the source code into an AST
    tree = ast.parse(source_code, filename=file_path)

    # Initialize the analyzer with the source code and the start and end lines for the analysis
    analyzer = DependencyAnalyzer(source_code, start_line, end_line, short_range, medium_range)

    analyzer.tree = tree  # Store the AST tree

    # analyzer.set_parent(tree)

    # Visit the AST to find usages and reason categories
    analyzer.visit(tree)

    # Perform the dependency analysis
    sorted_reason_categories_output, sorted_horizon_categories_output, reason_categories_printout, horizon_categories_printout = analyzer.analyze_dependencies(file_path, read_from_results=args.read_dependency_results, save_results=args.save_dependency_results)

    # # After analysis, you can access the results like this:
    # sorted_reason_categories_output, sorted_horizon_categories_output, reason_categories_printout, horizon_categories_printout = analyzer.analyze_dependencies()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze code dependencies.')
    parser.add_argument('file_path', type=str, help='Path to the source code file.')
    parser.add_argument('--short_range', type=int, default=5, help='Distance for Short-Range dependencies.')
    parser.add_argument('--medium_range', type=int, default=20, help='Distance for Medium-Range dependencies.')
    parser.add_argument('--read_dependency_results', action='store_true', default=False, help='Read dependency results from a CSV file.')
    parser.add_argument('--save_dependency_results', action='store_true', default=False, help='Save dependency results to a CSV file.')
    parser.add_argument('--update_def_line', action='store_true', help='Update the definition line of variables when value reassign.')
    args = parser.parse_args()

    main(args.file_path, args.short_range, args.medium_range)