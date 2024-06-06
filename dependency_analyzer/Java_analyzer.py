import os
import pandas as pd
import warnings
import json
import javalang
import argparse
import subprocess


class JavaDependencyAnalyzer:
    def __init__(self, source_code, start_line, end_line, short_range, medium_range, update_def_line=False):
        self.tree = javalang.parse.parse(source_code)
        self.start_line = start_line
        self.end_line = end_line
        self.short_range = short_range
        self.medium_range = medium_range
        self.inferred_variable_types = {}  # Maps variable names to their inferred types
        self.scope_stack = [{'type': 'global', 'locals': {}, 'classes': {}, 'interfaces': {}, 'methods': {}, 'libraries': {}, 'loops': {}, 'globals': {}, 'lambda':{}, 'stream':{}}]
        self.usages = []  # Holds variable and method usages
        self.horizon_categories = []  # Dependency details
        self.reason_categories = []  # Specific code constructs detected
        self.well_known_java_classes = {
                                        'System', 'PrintStream',
                                        'assertEquals', 'assertTrue', 'assertFalse',
                                        'Files', 'Paths', 'Objects', 
                                        'LinkedList', 'Set', 'HashSet', 'TreeSet', 
                                        'List', 'ArrayList', 'Arrays',
                                        'Map', 'HashMap', 'TreeMap'
                                        }
        self.well_known_static_members= {
                                            'System.out': 'java.io.PrintStream',
                                            'System.err': 'java.io.PrintStream',
                                        }
        self.collection_methods =       {'add', 'get', 'remove', 'clear', 'size', 'isEmpty', 'contains', 'toList', 'getValue', 'getKey', 'entrySet'}
        self.well_known_java_packages = {'java.io', 'java.net', 'java.text', 'java.time'}
        self.debug_print = False
        self.filter_none_def_line = True
        self.update_def_line = update_def_line
        self.report_super = True
        self.csv_save_path = './Analysis_Results/Dependency_Records/Java/'

    def current_scope(self):
        return self.scope_stack[-1]
    
    # Assuming you have enhanced scope entry to include more information
    def enter_scope(self, scope_type='general', name=None, inherited_globals=None, enclosing_class=None):
        new_scope = {
            'type': scope_type,
            'name': name,
            'locals': {},
            'classes': {},
            'interfaces': {},  # Add interfaces to the scope dictionary
            'methods': {},
            'libraries': {},
            'loops': {},
            'lambda': {},
            'stream': {},
            'globals': inherited_globals or {},
            'enclosing_class': enclosing_class
        }
        if self.debug_print:
            print(f"Entering scope: {new_scope}")
        self.scope_stack.append(new_scope)

    @staticmethod
    def call_java_class_existence_checker(base_package, potential_subclass):
        # Assuming the Java class files are in the same directory as this Python script
        command = ['java', '-cp', '.', 'dependency_analyzer.ClassExistenceChecker', base_package, potential_subclass]
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
            output = result.stdout.strip()
            # Assuming your Java program prints "true" for class existence and "false" otherwise
            return output.lower() == "true"
        except subprocess.CalledProcessError as e:
            print(f"Error calling Java program: {e}")
            return False

    def resolve_variable_scope(self, name, entity_name=None, inside_this=False): 
        # Initialize a variable to store the result of library checks to avoid redundancy
        library_result = None

        for scope in reversed(self.scope_stack):
            # When 'this' is used, ignore locals and directly look for globals.
            if inside_this:
                if name in scope.get('globals', {}):
                    return 'globals', scope['globals'][name]
                continue  # Skip checking locals when 'this' is use
            
            if entity_name:
                # Handling methods nested within a specified class or interface
                for cand_entity_name, methods in scope.get('methods', {}).items():
                    if isinstance(methods, dict) and name in methods and cand_entity_name == entity_name:
                        return 'methods', methods[name]
                continue  # Skip checking locals when class_name is provided

            # Check for local variables first
            if name in scope['locals']:
                return 'locals', scope['locals'][name]

            # Then check for global variables (instance fields)
            if name in scope['globals']:
                return 'globals', scope['globals'][name]

            # Check for methods, including those within classes and interfaces
            if name in scope['methods']:
                method_info = scope['methods'][name]
                # Check for nested class or interface methods
                if isinstance(method_info, dict): 
                    #Determine if the entity is a class or interface
                    if name in scope.get('classes', {}):
                        entity_line = scope['classes'][name]
                        entity_type = 'classes'
                        return entity_type, {'line': entity_line, 'is_instance': False}
                    elif name in scope.get('interfaces', {}):
                        entity_line = scope['interfaces'][name]
                        entity_type = 'interfaces'
                        return entity_type, {'line': entity_line, 'is_instance': False}
                else:  # Check for standalone methods 
                    return 'methods', method_info     
            
            # Handling nested class names and their methods
            for cand_entity_name, methods in scope.get('methods', {}).items():
                if isinstance(methods, dict) and name in methods:
                    return 'methods', methods[name]

            # Check classes
            if name in scope['classes']:
                return 'classes', {'line': scope['classes'][name], 'is_instance': True}
            
            # Add interfaces to the scope resolution
            if name in scope['interfaces']:
                return 'interfaces', scope['interfaces'][name]

            # Libraries check (deferred to avoid repeated checks)
            if library_result is None:  # Perform the check only if not already done
                library_result = self.check_library_imports(name, scope)
                if library_result:  # If a library result was found, return it
                    return 'libraries', library_result

            # Check for loops and lambdas
            if name in scope['loops']:
                return 'loops', scope['loops'][name]
            if name in scope['lambda']:
                return 'lambda', scope['lambda'][name]

        if self.debug_print:
            print("############### unresolved variable ###############: ", name)
        return 'unresolved', None  # If not found, return unresolved

    def check_library_imports(self, name, scope):
        """Check for direct library import or a match within wildcard import."""
        for lib_key, lib_info in scope['libraries'].items():
            # Direct import check
            if lib_key.split('.')[-1] == name:
                return lib_info['import_path']
            # Wildcard import check
            if lib_info['is_wildcard'] and self.call_java_class_existence_checker(lib_key.rstrip('.*'), name):
                return lib_info['import_path']
        return None

    def exit_scope(self):
        if len(self.scope_stack) > 1:
            exiting_scope = self.scope_stack.pop()
            enclosing_scope = self.current_scope()

            if self.debug_print:
                print(f"Exiting scope: {exiting_scope}")
            
            # When exiting a method scope, handle methods defined within classes or interfaces
            if exiting_scope['type'] == 'methods' and 'enclosing_class' in exiting_scope:
                method_name = exiting_scope['name']
                method_def_line = exiting_scope['methods'].get(method_name, 'unknown')
                parent_name = exiting_scope['enclosing_class']
                if parent_name:
                    # Ensure the parent (class or interface) exists in the enclosing scope's 'methods' dictionary
                    if parent_name not in enclosing_scope['methods']:
                        enclosing_scope['methods'][parent_name] = {}
                    # Update the parent's methods dictionary with the exited method's information
                    enclosing_scope['methods'][parent_name][method_name] = method_def_line

            # Handle class or interface scope exit
            if exiting_scope['type'] in ['classes']:

                entity_name = exiting_scope['name']
                if entity_name:
                    # Propagate entity definition line
                    entity_def_line = exiting_scope['classes'].get(entity_name, 'unknown')
                    enclosing_scope['classes'][entity_name] = entity_def_line

                    # Check if there are methods in the exiting entity scope to be propagated
                    if 'methods' in exiting_scope and entity_name in exiting_scope['methods']:
                        # Ensure the enclosing scope is prepared to hold methods for this entity
                        if entity_name not in enclosing_scope['methods']:
                            enclosing_scope['methods'][entity_name] = {}
                        
                        # Propagate each method of the entity
                        for method_name, method_line in exiting_scope['methods'][entity_name].items():
                                enclosing_scope['methods'][entity_name][method_name] = method_line
        else:
            print("Attempted to exit global scope - action prevented.")

    def add_variable(self, name, line, var_type, is_instance=False, class_type=None, update_def_line=False):
        # var_type might be 'locals', 'globals', 'classes', 'methods', or 'libraries'
        # Update the current scope with the new variable
        current_scope = self.current_scope()
        if self.debug_print:
            print(f"Adding {var_type} '{name}' at line {line} to scope {current_scope['type']}")
        if var_type not in current_scope:
            current_scope[var_type] = {}  # Create a new category if it doesn't exist
        ############################################################
        variable_info = current_scope[var_type].get(name, None)
        if variable_info and update_def_line:
            # Update def_line only if current line is after the original def_line
            # and if it's a reassignment or initialization in a different block.
            if isinstance(variable_info, dict) and 'line' in variable_info:
                if line > variable_info['line']:
                    variable_info['line'] = int(line)
            elif isinstance(variable_info, int):
                variable_info = max(variable_info, int(line))
            else:
                raise TypeError("variable_info must be a dict with a 'line' key or an int")
        else:
        ############################################################
            # First time variable is being added
            if var_type == 'locals':
                # Include an is_instance flag in the variable's record
                current_scope[var_type][name] = {'line': line, 'is_instance': is_instance, 'class_type': class_type}
            else:
                current_scope[var_type][name] = line

    def usages_append(self, name, usage_line, def_line, scope_type, usage_type):
        # Conditionally add usage based on filter_none_def_line and whether def_line is not None
        if not self.filter_none_def_line or (self.filter_none_def_line and def_line is not None):
            self.usages.append({
                'name': name,
                'usage_line': int(usage_line) if usage_line is not None else 'unknown',
                'def_line': int(def_line) if def_line is not None else 'unknown',
                'scope_type': scope_type if scope_type is not None else 'external or undefined',
                'type': usage_type
            })

    def add_usage(self, name, line, usage_type):
        def_line, scope_type = self.find_definition(name, usage_type)

        # If not found in the local scopes, attempt to resolve as a library usage
        if def_line is None:
            def_line, temp_scope_type = self.resolve_library_usage(name, line)
            scope_type = temp_scope_type if def_line is not None else 'unresolved'

        self.usages_append(name, line, def_line, scope_type, usage_type)
    
    def find_definition(self, name, usage_type):
        for scope in reversed(self.scope_stack):
            if usage_type == 'methods':
                def_line, scope_type  = self.find_method_definition(scope, name)
                if def_line and scope_type:
                    return def_line, scope_type
            else:
                if name in scope.get(usage_type, {}):
                    if usage_type == 'locals':
                        def_line = scope[usage_type][name].get('line')
                    else:
                        def_line = scope[usage_type].get(name)
                    return def_line, usage_type
        return None, None
    
    def find_method_definition(self, scope, name):
        # Directly defined methods
        if name in scope.get('methods', {}) and isinstance(scope['methods'].get(name), int):
            def_line = scope['methods'].get(name)
            scope_type = 'methods'
            return def_line, scope_type
        
        # Methods within classes: Adjusted to correctly handle dictionary structure
        for class_name, methods_dict in scope.get('methods', {}).items():
            if isinstance(methods_dict, dict) and name in methods_dict and isinstance(methods_dict[name], int):
                def_line = methods_dict[name]
                scope_type = f'methods'
                return def_line, scope_type
        return None, None

            
    def resolve_library_usage(self, name, line):
        """
        Attempt to resolve a usage as a library import, checking through the entire scope stack.
        Decides if it should be reported based on specific criteria.
        """
        for scope in reversed(self.scope_stack):  # Iterate through all scopes to find library imports
            for lib_key, lib_info in scope.get('libraries', {}).items():
                # Direct import check or wildcard import base package match
                if lib_key.split('.')[-1] == name or (lib_info['is_wildcard'] and JavaDependencyAnalyzer.call_java_class_existence_checker(lib_key.rstrip('.*'), name)):
                    if self.should_report_library(lib_key):
                    # if self.should_report_library(lib_key, lib_info, name):
                        # Use the stored 'def_line' for the library's definition line
                        return lib_info.get('def_line'), 'libraries'
        return None, None



    def should_report_library(self, import_path):
        #direct check for well-known classes
        if import_path in self.well_known_java_classes:
            return False

        if any(import_path.endswith('.' + cls) for cls in self.well_known_java_classes):
            return False

        if any(import_path.startswith(pkg) for pkg in self.well_known_java_packages):
            return False

        # Special handling for wildcard imports
        if import_path.endswith('*'):
            base_package = import_path[:-1]  # Remove the '*' to get the base package
            if any(base_package.startswith(pkg) for pkg in self.well_known_java_packages):
                return False

        return True

    def add_reason_category(self, reason, line):
        self.reason_categories.append({'reason_category': reason, 'usage_line': line})

    def is_builtin_method_call(self, method_invocation_node):
        # Utilize existing definitions for well-known classes and static members
        qualifier = getattr(method_invocation_node, 'qualifier', None)
        member = method_invocation_node.member
        
        # Direct checks for static members and collection methods
        if qualifier in self.well_known_java_classes or any(member in q for q in self.well_known_static_members):
            return True
        if member in self.collection_methods and self.is_collection_type(qualifier):
            return True
        ############################################
        # Leverage the should_report_library logic by constructing a hypothetical import path
        # from the qualifier and member. This is a simplification and might not always accurately
        # represent the actual import path or usage.
        hypothetical_import_path = f"{qualifier}.{member}" if qualifier else member
        
        # If should_report_library returns True, it means it's not a well-known library, thus not built-in
        # Invert the logic since is_builtin is True when it should NOT be reported (i.e., is well-known)
        return not self.should_report_library(hypothetical_import_path)

                
    def is_collection_type(self, qualifier):
        # Checks against both directly known types and those inferred during analysis
        known_collection_types = {'List', 'ArrayList', 'LinkedList', 'Set', 'HashSet', 'TreeSet', 'Map', 'HashMap', 'TreeMap', 'Paths'}
        # Initially check if the qualifier directly matches known collection types
        if qualifier in known_collection_types:
            return True

        # Check if the qualifier has been associated with a collection type based on previous variable declarations
        # This requires tracking variable types based on their initializations and assignments in the code.
        # For example, if 'column' was previously declared or assigned with 'new ArrayList<>()' or 'Arrays.asList()',
        # we should have recorded that 'column' is of type 'List'.
        #
        # This part assumes your code analysis includes tracking such type information. If it does not,
        # implementing full type tracking would require significant changes beyond this function.
        # Here, I'm showing a simplified approach.

        # Assume a mapping from variable names to their inferred types exists
        inferred_type = self.inferred_variable_types.get(qualifier, '')
        return inferred_type in known_collection_types


    def is_called_on_collection_without_qualifier(self, method_invocation_node):
        # This is a placeholder for a more advanced analysis, where you'd check the type of the variable
        # on which a method like 'get' or 'add' is called, without a direct qualifier in the method call.
        # This would typically involve tracking the types of local variables, parameters, and fields.
        return False  # Implementing this accurately requires context-specific analysis beyond simple checks.

    def get_do_statement_line_number(self, node):
        """Get the line number for a do-while loop's condition."""
        # Typically, the 'while' part of a do-while loop does not have its own line number.
        # We use the condition's line number as a proxy.
        # First, attempt to use the condition's position if available.
        line_number = self.get_line_number_from_node(node.condition) if self.get_line_number_from_node(node.condition) else 'unknown'

        # Fallback: if the line number is still unknown, approximate using the body's last statement.
        if line_number == 'unknown' and node.body and node.body.statements:
            last_statement = node.body.statements[-1]
            # Assuming the last statement's position is indicative of the loop's range.
            line_number = getattr(last_statement.position, 'line', 'unknown') + 1  # Approximate 'while' as following the last statement
        return line_number

    def handle_for_control(self, node, loop_line):
        # Handling traditional for-loop
        if isinstance(node.control, javalang.tree.ForControl):
            if node.control.init:
                for init_part in node.control.init:
                    # Handling a tuple structure in init_part
                    # We're interested in the VariableDeclaration part
                    for item in init_part:
                        if isinstance(item, javalang.tree.VariableDeclaration):
                            # Now iterate through the declarators to extract variable names
                            for declarator in item.declarators:
                                self.add_variable(declarator.name, loop_line, 'locals')
                                # If the initializer of the variable is another variable, record its usage.
                                if declarator.initializer:
                                    # Use walk_tree to handle any node type in the initializer
                                    self.walk_tree(declarator.initializer, parent=node, line_number=loop_line)
                        else:
                            self.walk_tree(item, parent=node, line_number=loop_line)

        # Enhanced for-loop variable handling
        elif isinstance(node.control, javalang.tree.EnhancedForControl):
            # In enhanced for-loops, there's typically one variable being declared to iterate over
            # Ensure we have a valid variable and it's properly declared
            if node.control.var:
                for declarator in node.control.var.declarators:  # Handle each declarator in enhanced for-loop
                    self.add_variable(declarator.name, loop_line, 'locals')
                    if node.control.iterable:
                        self.walk_tree(node.control.iterable, parent=node, line_number=loop_line)

        
        if node.control:
            if hasattr(node.control, 'condition') and node.control.condition:
                self.walk_tree(node.control.condition, parent=node, line_number=loop_line)  # Handle the loop condition
            # Now handle the initialization and update parts of the loop
            if hasattr(node.control, 'init') and node.control.init:
                for init in node.control.init:
                    self.walk_tree(init, parent=node, line_number=loop_line)
            if hasattr(node.control, 'update') and node.control.update:
                for update in node.control.update:
                    self.walk_tree(update, parent=node, line_number=loop_line)

        # Now process the body of the loop to ensure any usages are captured
        if node.body:
            self.walk_tree(node.body, parent=node, line_number=loop_line)

    def process_child_nodes(self, node, line_number=None):
        """Recursively process child nodes of the current node."""
        if hasattr(node, 'children'):
            for child in node.children:
                if isinstance(child, list):
                    for item in child:
                        if isinstance(item, javalang.tree.Node) and not getattr(item, 'already_processed', False):
                            self.walk_tree(item, parent=node, line_number=line_number)
                elif isinstance(child, javalang.tree.Node) and not getattr(child, 'already_processed', False):
                    self.walk_tree(child, parent=node, line_number=line_number)
        
    def find_class_method_def_line(self, class_name, method_name):
        # Iterate through the scope stack from top (most local scope) to bottom (global scope)
        for scope in reversed(self.scope_stack):
            # Check if the current scope or any enclosing scope has the class and its methods
            if class_name in scope['classes']:
                class_methods = scope['methods'].get(class_name, {})
                if method_name in class_methods:
                    return class_methods[method_name]
        return 'unknown'
    
    def check_qualifier(self, node, qualifier):
        # Resolve the scope and type of the qualifier
        qualifier_type, qualifier_info = self.resolve_variable_scope(qualifier)
        if qualifier_type == 'classes':
            #Handling class method invocation by combining qualifier and member
            method_def_line = self.find_class_method_def_line(qualifier, node.member)
            self.usages_append(qualifier + '.' + node.member, node.position.line if node.position else 'unknown', method_def_line, 'methods', 'methods')
        
        elif qualifier_type == 'locals' and qualifier_info and qualifier_info.get('is_instance', False):
            class_type = qualifier_info.get('class_type', None)
            if class_type:
                # Use the stored class type to find the method definition line
                method_def_line = self.find_class_method_def_line(class_type, node.member)
                if method_def_line != 'unknown':
                    self.usages_append(f"{class_type}.{node.member}", node.position.line if node.position else 'unknown', method_def_line, 'methods', 'methods')
                else:
                    self.add_usage(qualifier, node.position.line if node.position else 'unknown', qualifier_type)
        elif qualifier_type in ['libraries'] and self.should_report_library(qualifier):
            # Add the usage if the qualifier is recognized as a variable or method within scope
            self.add_usage(qualifier, node.position.line if node.position else 'unknown', qualifier_type)

        elif qualifier_type in ['locals', 'globals', 'methods']:
            # Add the usage if the qualifier is recognized as a variable or method within scope
            self.add_usage(qualifier, node.position.line if node.position else 'unknown', qualifier_type)

    def pre_collect_class_and_methods(self, node):
        if isinstance(node, javalang.tree.CompilationUnit):
            for type_decl in node.types:
                if isinstance(type_decl, javalang.tree.ClassDeclaration):
                    # Initialize class information with line number directly as value
                    self.scope_stack[0]['classes'][type_decl.name] = type_decl.position.line if type_decl.position else 'unknown'
                    
                    # Initialize an empty dictionary for methods under the global scope 'methods'
                    self.scope_stack[0]['methods'][type_decl.name] = {}
                    
                    # Collect methods
                    for member in type_decl.body:
                        # if isinstance(member, javalang.tree.MethodDeclaration) or isinstance(member, javalang.tree.ConstructorDeclaration):
                        if isinstance(member, javalang.tree.MethodDeclaration):
                            # Place method line numbers directly as values under the class's entry in 'methods'
                            self.scope_stack[0]['methods'][type_decl.name][member.name] = member.position.line if member.position else 'unknown'
                elif isinstance(type_decl, javalang.tree.InterfaceDeclaration):
                    # Similar handling for interfaces
                    self.scope_stack[0]['interfaces'][type_decl.name] = type_decl.position.line if type_decl.position else 'unknown'
                    self.scope_stack[0]['methods'][type_decl.name] = {}
                    
                    for member in type_decl.body:
                        if isinstance(member, javalang.tree.MethodDeclaration):
                            # Store interface methods similarly
                            self.scope_stack[0]['methods'][type_decl.name][member.name] = member.position.line if member.position else 'unknown'

    def handle_regular_method_invocation(self, node):
        """Handle non-stream method invocations."""
        qualifier = getattr(node, 'qualifier', None)
        if qualifier:
            self.check_qualifier(node, qualifier)
        else:
            # Determine if the method invocation is a built-in method
            is_builtin_method = self.is_builtin_method_call(node)
            if not is_builtin_method:
                var_scope, var_info = self.resolve_variable_scope(node.member)
                # Add the method invocation to usages if it's not recognized as built-in
                self.add_usage(node.member, node.position.line if node.position else 'unknown', var_scope)

    def handle_member_reference(self, node, line_number=None, inside_this=False, update_def_line=False):
        """
        Handle a MemberReference node by determining its usage and adding it to the usages list.
        If update_def_line is True, update the variable's definition line.
        """
        line_number = line_number if line_number else self.get_line_number_from_node(node)
        qualifier = getattr(node, 'qualifier', None)
        if qualifier:
            self.check_qualifier(node, qualifier)
        else:
            # Resolve the member without a qualifier
            usage_type, def_line = self.resolve_variable_scope(node.member, inside_this=inside_this)
            # Add usage with the current scope resolution and line number
            self.add_usage(node.member, line_number, usage_type)
            #If update_def_line is True, update the variable's definition line
            if update_def_line:
                self.add_variable(node.member, line_number, usage_type, update_def_line=True)
                node.already_processed = True
        if node.selectors:
            # Handle any selectors that might be part of the MemberReference
            for selector in node.selectors:
                # Recursively process each selector which could also be a MemberReference or other node types
                if isinstance(selector, javalang.tree.MemberReference):
                    self.handle_member_reference(selector, line_number, inside_this=inside_this, update_def_line=update_def_line)
                else:
                    self.walk_tree(selector, node, line_number)
                    


    def handle_local_variable_declaration(self, node):
        """
        Handle local variable declarations and check if the type refers to a library.
        """
        for declarator in node.declarators:
            if node.position:
                # Determine if the variable is an instance variable
                is_class_instance = isinstance(declarator.initializer, javalang.tree.ClassCreator)
                # Store the class type for instance variables
                class_type = declarator.initializer.type.name if is_class_instance else None
                self.add_variable(declarator.name, node.position.line, 'locals', is_instance=is_class_instance, class_type=class_type)
                self.check_and_add_type_usage(node.type.name, node.position.line)
                if isinstance(node.type, javalang.tree.ReferenceType):
                    self.handle_generic_types(node.type, node.position.line)
                # Analyze the initializer for expressions involving library types
                if declarator.initializer:
                    self.walk_tree(declarator.initializer, node, node.position.line)

    def handle_parameter_declaration(self, parameters):
        """
        Handle parameter declarations in methods or constructors and check if the type refers to a library.
        """
        for param in parameters:
            if param.position:
                self.add_variable(param.name, param.position.line, 'locals')
                self.check_and_add_type_usage(param.type.name, param.position.line)
                if isinstance(param.type, javalang.tree.ReferenceType):
                    self.handle_generic_types(param.type, param.position.line)

                # Handle default value initializers for parameters (if any)
                if hasattr(param, 'initializer') and param.initializer:
                    self.walk_tree(param.initializer, parent=param, line_number=param.position.line)

    def check_and_add_type_usage(self, type_name, line):
        """
        Check if a given type name refers to a library import and record its usage.
        """
        scope_type, scope_info = self.resolve_variable_scope(type_name)
        if scope_type in ['libraries'] and self.should_report_library(type_name):
            self.add_usage(type_name, line, scope_type)
        elif scope_type in ['classes', 'methods']:
            self.add_usage(type_name, line, scope_type)
    
    def handle_if_statement(self, node, line_number):
        if node.condition:
            self.walk_tree(node.condition, parent=node, line_number=line_number)
        if node.then_statement:
            self.walk_tree(node.then_statement, parent=node, line_number=line_number)
        if node.else_statement:
            self.walk_tree(node.else_statement, parent=node, line_number=line_number)

    def handle_generic_types(self, type_node, line):
        """
        Recursively handle generic type arguments to identify library usages.
        """
        if hasattr(type_node, 'arguments') and type_node.arguments:
            for arg in type_node.arguments:
                if isinstance(arg, javalang.tree.ReferenceType):
                    self.check_and_add_type_usage(arg.name, line)
                    self.handle_generic_types(arg, line)  # Recursively handle nested generic types
                    
                elif isinstance(arg, javalang.tree.TypeArgument):
                    # Check if the argument has a type and it's not just a wildcard
                    if arg.type:
                        self.check_and_add_type_usage(arg.type.name, line)
                        self.handle_generic_types(arg.type, line)  # Recursively handle nested type arguments
                    # # Handle bounds on the type argument
                    # if arg.pattern:
                    #     if arg.pattern == 'extends' and arg.bound:
                    #         self.check_and_add_type_usage(arg.bound.name, line)
                    #     elif arg.pattern == 'super' and arg.bound:
                    #         self.check_and_add_type_usage(arg.bound.name, line)           
    
    def get_line_number_from_node(self, node, visited=None):
        """
        Recursively get the first available position from the node or its children, avoiding infinite recursion.
        """
        if visited is None:
            visited = set()

        # Check if we've already visited this node to prevent cycles
        if id(node) in visited:
            return None
        visited.add(id(node))
            
        # Check directly if the node has a position
        if hasattr(node, 'position') and node.position:
            return node.position.line
        
        # If the node is iterable (handles lists, sets, etc.), check its elements
        if hasattr(node, '__iter__') and not isinstance(node, str):
            for item in node:
                line = self.get_line_number_from_node(item, visited)
                if line is not None:
                    return line

        # Recursively check attributes of the node if they might be other nodes or lists
        if hasattr(node, '__dict__'):
            for key, value in node.__dict__.items():
                if isinstance(value, (javalang.tree.Node, list, tuple)):
                    line = self.get_line_number_from_node(value, visited)
                    if line is not None:
                        return line

        return None  # Return None if no position found

    def walk_tree(self, node, parent=None, line_number=None):
        # Check if node is a tuple or list and handle its elements
        if isinstance(node, (tuple, list)):
            for item in node:
                if isinstance(item, javalang.tree.Node):
                    self.walk_tree(item, parent, line_number)
            return
        
        # First check if node has been processed to skip if necessary
        if getattr(node, 'already_processed', False):
            return

        # Mark this node as processed to prevent future redundant processing
        node.already_processed = True

        # line_number is None, use the passed line_number 
        line_number = self.get_line_number_from_node(node) or line_number
        # If line_number is still None, return immediately to avoid processing without line number context
        if line_number is None:
            return
        
        # Handling lambda expressions
        if isinstance(node, javalang.tree.LambdaExpression):
            # Capture the line number of the lambda expression for reporting
            # Add a reason category for lambda usage
            self.add_reason_category('Lambda_Expressions', line_number)
            # Enter a new scope specifically for the lambda
            self.enter_scope('lambda', 'lambda_expression')
            # Register lambda parameters as local variables within this scope
            for param in node.parameters:
                self.add_variable(param.member, line_number, 'locals')

        # Handling specific stream operations within method invocations
        elif isinstance(node, javalang.tree.MethodInvocation):
            if node.member in ['stream', 'map', 'filter', 'collect'] and node.position:
                # Handling as a stream operation
                self.add_reason_category('Stream_Operations', line_number)
            else:
                # Regular method invocations
                self.handle_regular_method_invocation(node)
            # Process each argument in the method call
            for arg in node.arguments:
                self.walk_tree(arg, node, line_number)

        # Handling do-while loops
        elif isinstance(node, javalang.tree.DoStatement):
            line_number = self.get_do_statement_line_number(node)
            self.add_reason_category('Define Stop Criteria',line_number)

        # Handling if-else reasoning
        elif isinstance(node, javalang.tree.IfStatement):
            if node.position:
                self.add_reason_category('If-else Reasoning', line_number)
            # Now handle the if statement with proper scope management
            self.handle_if_statement(node, line_number)
        
        # Handling while loops - added logic
        elif isinstance(node, javalang.tree.WhileStatement):
            self.add_reason_category('Define Stop Criteria', line_number)

        # Handling traditional for-loops
        elif isinstance(node, javalang.tree.ForStatement):
            self.enter_scope('loops')
            self.add_reason_category('Define Stop Criteria', line_number)
            self.handle_for_control(node, line_number)

        # Handling try-catch-finally blocks
        elif isinstance(node, javalang.tree.TryStatement):
            # Process the 'try' block
            if node.block:
                self.walk_tree(node.block, parent=node, line_number=line_number)

            # Process each 'catch' clause
            for catch in node.catches:
                if catch.block:
                    self.walk_tree(catch.parameter, parent=catch, line_number=line_number)
                    self.walk_tree(catch.block, parent=catch, line_number=line_number)

            # Process the 'finally' block if it exists
            if node.finally_block:
                self.walk_tree(node.finally_block, parent=node, line_number=line_number)

        # Member references could be variables or class attributes
        elif isinstance(node, javalang.tree.MemberReference):
            # line_number = self.get_line_number_from_node(node)
            self.handle_member_reference(node)

        elif isinstance(node, javalang.tree.InterfaceDeclaration):
            # Entering a new interface scope
            self.enter_scope('interfaces', node.name)
            if node.position:
                # Register the interface in the current scope
                self.add_variable(node.name, line_number, 'interfaces')
            # Process each member of the interface
            for member in node.body:
                self.walk_tree(member, parent=node, line_number=line_number)

        # Handling ClassDeclaration nodes
        elif isinstance(node, javalang.tree.ClassDeclaration):
            # Entering a new class scope
            self.enter_scope('classes', node.name)
            if node.position:
                self.add_variable(node.name, line_number, 'classes')
                # Check if this class extends another and handle the superclass
                superclass_names, interface_names = self.handle_class_inheritance_and_interfaces(node, line_number)
                for member in node.body:
                    # Update line_number for each member
                    member_line_number = self.get_line_number_from_node(member) or line_number
                    if isinstance(member, javalang.tree.MethodDeclaration):
                        # Check for method overrides in the context of the current class scope
                        self.check_for_method_overrides(member, superclass_names, interface_names, member_line_number)
                        # Now handle the method declaration itself
                        self.handle_method_declaration(member, node)
                    else:
                        # Handle other members like fields or nested classes/interfaces
                        self.walk_tree(member, parent=node, line_number=member_line_number)

        # Handling constructor calls in object creation
        elif isinstance(node, javalang.tree.ClassCreator):
            # This is a direct instantiation of a class, e.g., new ArrayList
            self.check_and_add_type_usage(node.type.name, line_number)
            # Handle generic types in class instantiation
            if node.type and hasattr(node.type, 'arguments') and node.type.arguments:
                for arg in node.type.arguments:
                    if isinstance(arg, javalang.tree.ReferenceType):
                        self.check_and_add_type_usage(arg.name, line_number)
                        self.handle_generic_types(arg, line_number)

        elif isinstance(node, javalang.tree.MethodDeclaration) or isinstance(node, javalang.tree.ConstructorDeclaration):            
            # Directly handle Method and Constructor Declarations outside of class/interface contexts
            self.handle_method_declaration(node, parent)

        # Handling FieldDeclaration nodes
        elif isinstance(node, javalang.tree.FieldDeclaration):
            for declarator in node.declarators:
                if node.position:
                    self.add_variable(declarator.name, line_number, 'globals')

        # Handling VariableDeclarator nodes
        elif isinstance(node, javalang.tree.VariableDeclarator):
            if node.position and isinstance(parent, javalang.tree.LocalVariableDeclaration):
                self.add_variable(node.name, line_number, 'locals')

        # Adjust handling for relevant nodes
        elif isinstance(node, javalang.tree.LocalVariableDeclaration):
            self.handle_local_variable_declaration(node)

        elif isinstance(node, javalang.tree.Import):
            import_path = node.path
            # Attempt to infer wildcard usage; not recommended for precise parsing
            is_wildcard = node.wildcard
            # Capture the line number of the import statement
            def_line = line_number if line_number else None
            lib_info = {
                'is_wildcard': is_wildcard,
                'import_path': import_path,  # Store the full import path for later analysis
                'def_line': def_line  # Store the definition line of the import
            }
            self.current_scope()['libraries'][import_path] = lib_info
            if self.debug_print:
                print(f"Adding Library '{lib_info}' at line {def_line} to scope")
        
        # Handling binary operations
        elif isinstance(node, javalang.tree.BinaryOperation):
            # Process the left operand
            if node.operandl:
                self.walk_tree(node.operandl, parent=node, line_number=self.get_line_number_from_node(node.operandl))

            # Process the right operand
            if node.operandr:
                self.walk_tree(node.operandr, parent=node, line_number=self.get_line_number_from_node(node.operandr))

        elif isinstance(node, javalang.tree.This):
            self.handle_this_reference(node, line_number)
            # Specific handling for 'This' references

        # Handling assignments
        elif isinstance(node, javalang.tree.Assignment):
            # Attempt to get the line number directly from the node, or from the left-hand side if unavailable
            line_number = self.get_line_number_from_node(node)
            # This assumes simple assignments of the form 'a = ...'
            # For assignments like 'this.a = ...', you would need to handle them differently

            # Handling the right-hand side first
            if node.value:
                self.walk_tree(node.value, node, line_number)

            if node.expressionl:
                if isinstance(node.expressionl, javalang.tree.MemberReference):
                    self.handle_member_reference(node.expressionl, line_number, update_def_line=self.update_def_line)

                # Handle 'This' with potentially nested MemberReferences
                elif isinstance(node.expressionl, javalang.tree.This):
                    for selector in node.expressionl.selectors:
                        if isinstance(selector, javalang.tree.MemberReference):
                            self.handle_member_reference(selector, line_number, inside_this=True, update_def_line=self.update_def_line)
                        else:
                            self.walk_tree(selector, node.expressionl, line_number)
                else:
                    # General case for other types of expression
                    self.walk_tree(node.expressionl, node, line_number)

        self.process_child_nodes(node, line_number=line_number)
        # Conditionally exit scope; MethodDeclaration and ConstructorDeclaration nodes are handled separately
        if isinstance(node, (
                            javalang.tree.LambdaExpression,
                            javalang.tree.ForStatement,
                            javalang.tree.InterfaceDeclaration,
                            javalang.tree.ClassDeclaration,
                            )):
            self.exit_scope()
    
    def handle_this_reference(self, node, line_number=None):
        # This node handling logic, possibly handling selectors or method calls
        if node.selectors:
            for selector in node.selectors:
                if isinstance(selector, javalang.tree.MemberReference):
                    self.handle_member_reference(selector, line_number, inside_this=True)
                else:
                    self.walk_tree(selector, node, line_number)
    
    def handle_class_inheritance_and_interfaces(self, class_node, class_line):
        superclass_names = []
        interface_names = []

        # Handle inheritance (superclass)
        if class_node.extends:
            if isinstance(class_node.extends, javalang.tree.ReferenceType):
                superclass_names.append(class_node.extends.name)
                self.add_usage(class_node.extends.name, class_line, 'classes')
            elif isinstance(class_node.extends, list):
                for extend in class_node.extends:
                    if isinstance(extend, javalang.tree.ReferenceType):
                        superclass_names.append(extend.name)
                        self.add_usage(extend.name, class_line, 'classes')

        # Handle interfaces implemented by the class
        if class_node.implements:
            for implement in class_node.implements:
                if isinstance(implement, javalang.tree.ReferenceType):
                    interface_names.append(implement.name)
                    self.add_usage(implement.name, class_line, 'interfaces')

        return superclass_names, interface_names
    
    def check_for_method_overrides(self, method_node, superclass_names, interface_names, method_line):
        # Check for overrides in superclass
        for superclass_name in superclass_names:
            if superclass_name:  # Only check for overrides if there's a superclass
                superclass_method_info, superclass_method_line = self.resolve_variable_scope(method_node.name, entity_name=superclass_name)
                if superclass_method_line != 'unknown' and superclass_method_line:
                    self.usages_append(method_node.name, method_line, superclass_method_line, superclass_method_info, 'methods')

        # Check for overrides in implemented interfaces
        for interface_name in interface_names:
            if interface_name:  # Only check for overrides if there's an interface
                interface_method_info, interface_method_line = self.resolve_variable_scope(method_node.name, entity_name=interface_name)
                if interface_method_line != 'unknown' and interface_method_line:
                    self.usages_append(method_node.name, method_line, interface_method_line, interface_method_info, 'methods')

    def handle_method_declaration(self, method_node, parent_node):
        #initializing
        entity_name = parent_node.name if isinstance(method_node, javalang.tree.ConstructorDeclaration) else method_node.name
        method_line = self.get_line_number_from_node(method_node)
        
        # Enter method or constructor scope
        self.enter_scope('methods', entity_name, enclosing_class=parent_node.name if parent_node and isinstance(parent_node, javalang.tree.ClassDeclaration) else None)
        self.add_variable(entity_name, method_line, 'methods')

        # Handling parameters
        self.handle_parameter_declaration(method_node.parameters)

        # Process the method body
        if method_node.body:
            for statement in method_node.body:
                if isinstance(statement, javalang.tree.StatementExpression):
                    if self.report_super and isinstance(statement.expression, javalang.tree.SuperConstructorInvocation):
                        # Handle the super() call
                        self.handle_super_call(parent_node, statement.expression.arguments, method_line)
                self.walk_tree(statement, parent=method_node, line_number=method_line)
        self.process_child_nodes(method_node, line_number=method_line)
        self.exit_scope()

    def handle_super_call(self, class_node, arguments, line_number):
        if class_node.extends:
            superclass_name = class_node.extends.name
            # print(f"Super call to {superclass_name} with arguments at line {line_number}")
            # Attempt to find the matching superclass constructor
            super_constructor_line = self.match_super_constructor(superclass_name, arguments)
            if super_constructor_line != 'unknown':
                self.add_reason_category('Super_Call', line_number)

    def match_super_constructor(self, superclass_name, arguments):
        # Retrieve constructors from the scope stack or external metadata
        constructors = self.get_constructors(superclass_name)
        for constructor_name, constructor_info in constructors.items():
            if isinstance(constructor_info, int):
                return constructor_info
        return 'unknown'

    def get_constructors(self, class_name):
        # Iterate over the scope stack in reverse to find the most relevant class constructor definitions
        for scope in reversed(self.scope_stack):
            if 'methods' in scope and class_name in scope['methods']:
                return scope['methods'][class_name]
        return {}

    
    def get_range_type(self, use_line, def_line):
        if def_line is None or use_line is None:
            return 'N/A'  # Not applicable if def_line is None
        # if type(use_line) != int or type(def_line) != int:
        #     import pdb; pdb.set_trace()
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
            var_name = result['name']
            var_line = result['usage_line']
            scope_type = result['scope_type']
            def_line = result['def_line']
            range_type = result['range_type']

            # # Apply similar skipping logic as in your Python dependency analyzer
            # if var_line == def_line and scope_type not in ['loops', 'list_comp', 'generator_exp', 'lambda']:
            #     continue
            if scope_type == 'builtin':
                continue

            # Constructing messages according to the scope type
            if scope_type == 'methods':
                message = f"Function '{var_name}' used at line {var_line}"
            elif scope_type == 'classes':
                message = f"Class '{var_name}' used at line {var_line}"
            elif scope_type == 'libraries':
                message = f"Library '{var_name}' used at line {var_line}"
            elif scope_type == 'list_comp':
                message = f"Variable '{var_name}' used at line {var_line} is part of a List_Comprehension"
            elif scope_type == 'generator_exp':
                message = f"Variable '{var_name}' used at line {var_line} is part of a Generator_Expression"
            elif scope_type == 'lambda':
                message = f"Variable '{var_name}' used at line {var_line} is part of a Lambda_Expression"
            elif scope_type == 'loops':
                message = f"Variable '{var_name}' used at line {var_line} is part of a Loop"
            elif scope_type == 'globals':
                message = f"Global_Variable '{var_name}' used at line {var_line}"
            elif scope_type == 'locals':
                message = f"Variable '{var_name}' used at line {var_line}"
            elif scope_type == 'interfaces':
                message = f"Interface '{var_name}' used at line {var_line}"

            # Append definition and dependency details
            def_part = f"is defined at line {def_line}" if def_line != 'external or undefined' else "is undefined"
            message += f" {def_part} and has a {range_type} dependency."

            # Add the constructed message to the output list
            horizon_categories_output.append(message)

        return horizon_categories_output

    def report_reason_analysis_results(self):
        # Store reason categories in a list
        reason_categories_output = []
        for result in self.reason_categories:
            reason = result['reason_category']
            lineno = int(result['usage_line'])
            # Check if the variable usage is within the analysis range
            if (self.start_line <= lineno <= self.end_line):
                reason_categories_output.append(f"'{reason}' detected at line {lineno}.")
        return reason_categories_output
    
    def print_list_elements(self, my_list):
        for element in my_list:
            print(element)

    def post_analysis(self):
        # Collect data without calculating range_type
        intermediate_dependency_results = []
        for usage in self.usages:
            # Here, 'def_line' and 'scope_type' have already been determined during the walk
            name, use_line, def_line, scope_type = usage['name'], usage['usage_line'], usage['def_line'], usage['scope_type']
            
            # Check if the variable usage is within the analysis range
            if (self.start_line <= use_line <= self.end_line):
                # Record the usage with its immediate definition info, if found
                intermediate_dependency_results.append({
                    'name': name,
                    'usage_line': int(use_line),
                    'def_line': int(def_line) if def_line != 'unknown' else 'external or undefined',
                    'scope_type': scope_type if scope_type != 'unresolved' else 'external or undefined',
                })
        
        # Sort the list of dictionaries by 'usage_line'
        self.horizon_categories = sorted(intermediate_dependency_results, key=lambda x: int(x['usage_line']))
        # Remove duplicate dictionaries from the list
        self.horizon_categories = self.remove_duplicate_dict(self.horizon_categories)
        # Sorting the list of dictionaries by 'usage_line'
        self.reason_categories = sorted(self.reason_categories, key=lambda x: int(x['usage_line']))
        # Remove duplicate dictionaries from the list
        self.reason_categories = self.remove_duplicate_dict(self.reason_categories)
    
    def update_dependency_range(self, intermediate_dependency):
        # Update the range_type for each record
        updated_dependency_results = []
        for item in intermediate_dependency:
            use_line = int(item['usage_line'])
            def_line = int(item['def_line']) if item['def_line'] != 'external or undefined' else None
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
            # Pre-collect class and method information
            self.pre_collect_class_and_methods(self.tree)
            self.walk_tree(self.tree)
            if self.debug_print:
                # Analysis code here: populate usages, horizon_categories, reason_categories based on walked data
                print("Final Scope Stack:", self.scope_stack)# Print the final scope stack
                print("Usages Recorded:", self.usages)  # Print all recorded usages
            
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


def main(args, file_path, short_range, medium_range):
    # Read the source code from file
    with open(file_path, 'r') as file:
        source_code = file.read()
    # Define the markers
    start_marker = "//######### Analyze from here #########\n"
    end_marker = "//######### Analyze End here #########\n"

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


    analyzer = JavaDependencyAnalyzer(source_code, start_line, end_line, short_range, medium_range, update_def_line=args.update_def_line)
    analyzer.analyze_dependencies(file_path, read_from_results=args.read_dependency_results, save_results=args.save_dependency_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Java code dependencies.')
    parser.add_argument('file_path', type=str, help='Path to the Java source code file.')
    parser.add_argument('--short_range', type=int, default=5, help='Short-range dependency distance.')
    parser.add_argument('--medium_range', type=int, default=20, help='Medium-range dependency distance.')
    parser.add_argument('--read_dependency_results', action='store_true', default=False, help='Read dependency results from a CSV file.')
    parser.add_argument('--save_dependency_results', action='store_true', default=False, help='Save dependency results to a CSV file.')
    parser.add_argument('--update_def_line', action='store_true', help='Update the definition line of variables when value reassign.')
    args = parser.parse_args()
    main(args, args.file_path, args.short_range, args.medium_range)