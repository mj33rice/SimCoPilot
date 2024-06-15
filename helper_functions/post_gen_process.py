from openai import OpenAI
from difflib import SequenceMatcher
from collections import Counter
import ast
import re
import unittest

def get_indent_char(before_lines, indent_check_lines=10):
    # Check the indentation size of a sample of lines from 'before_lines'
    indent_sizes = []
    for i in range(len(before_lines)-1, -1, -1):
        line = before_lines[i]
        match = re.match(r'^(\s*)', line)
        indent = match.group(1) if match else ''
        indent_sizes.append(len(indent))
        if len(indent_sizes) >= indent_check_lines:
            break
    # If '\t' is used for indentation, return '\t'
    if '\t' in before_lines:
        return '\t'

    # Calculate relative indentations
    relative_indents = [abs(j-i) for i, j in zip(indent_sizes[:-1], indent_sizes[1:])]
    # Use the most common relative indentation size as the indent size
    # If there's only one element in indent_sizes, use that as the indent size
    if len(indent_sizes) == 1:
        indent_size = indent_sizes[0]
    elif relative_indents:
        relative_indents = [indent for indent in relative_indents if indent != 0]
        if relative_indents:
            indent_size = Counter(relative_indents).most_common(1)[0][0]
        else:
            indent_size = 0

    # If the most common relative indentation size is 0, return '\t'
    if indent_size == 0:
        return '\t'

    # Set the indentation character based on the indent size
    indent_char = ' ' * indent_size

    return indent_char

def calculate_indentation(line, indent_char, tab_indent=4):
    """Calculate the indentation level of a line."""
    line_with_spaces = line.replace('\t', ' ' * tab_indent) if indent_char == '\t' else line
    return len(line_with_spaces) - len(line_with_spaces.lstrip())
####################
def remove_extra_brackets(between_lines, bracket, num_brackets_to_remove):
    # Split the lines into a list
    lines = between_lines.split('\n')

    # Iterate over the lines in reverse order
    for i in range(len(lines) - 1, -1, -1):
        # If the line ends with the bracket, remove it and decrease the count
        if lines[i].endswith(bracket):
            lines[i] = lines[i][:-1]
            num_brackets_to_remove -= 1

        # If we have removed all the necessary brackets, stop
        if num_brackets_to_remove == 0:
            break

    # Join the lines back into a string
    between_lines = '\n'.join(lines)

    return between_lines

def auto_bracket_matcher(before_lines, between_lines, after_lines):
    # Define the pairs of brackets
    brackets = {'{': '}', '[': ']', '(': ')'}

    before_lines = '\n'.join(before_lines)
    after_lines = '\n'.join(after_lines)
    
    # Concatenate the lines
    total_lines = before_lines + between_lines + after_lines

    # Initialize a dictionary to count the brackets
    bracket_counts = {bracket: [0, 0] for bracket in brackets}

    # Count the number of open and close brackets
    for char in total_lines:
        if char in brackets:
            bracket_counts[char][0] += 1
        elif char in brackets.values():
            for bracket, close_bracket in brackets.items():
                if char == close_bracket:
                    bracket_counts[bracket][1] += 1

    # Add missing closing brackets to the end of between_lines
    for bracket, counts in bracket_counts.items():
        open_brackets, close_brackets = counts
        if open_brackets > close_brackets:
            between_lines += brackets[bracket] * (open_brackets - close_brackets)
        elif open_brackets < close_brackets:
            # Remove extra closing brackets from the end of between_lines
            between_lines = remove_extra_brackets(between_lines, brackets[bracket], close_brackets - open_brackets)
    return between_lines

def auto_indent(before_lines, between, after_lines, tab_indent=4):
    """
    Clean the generated 'between' code and merge it with 'before' and 'after' parts.

    Parameters:
    before_lines (list[str] or str): Lines of code before the 'between' part.
    between (str): The generated 'between' part of the code.
    after_lines (list[str] or str): Lines of code after the 'between' part.

    Returns:
    str: Cleaned and merged code.
    """
    # Define the counterparts for each keyword
    counterparts = {
        'elif ': ['elif ', 'if '],
        'else:': ['elif ', 'if '],
        'except ': ['try '],
        'finally:': ['try '],
    }
    # Ensure before_lines and after_lines are lists of strings
    if isinstance(before_lines, str):
        before_lines = before_lines.split('\n')
    if isinstance(after_lines, str):
        after_lines = after_lines.split('\n')

    #Detect the indentation style (spaces or tabs) of the 'before' part
    indent_char = get_indent_char(before_lines)
    
    # Determine the indentation level of the last non-empty line in 'before_lines'
    last_non_empty_line = next((line for line in reversed(before_lines) if line.strip()), "")
    last_before_line_indent = calculate_indentation(last_non_empty_line, indent_char, tab_indent=tab_indent)

    # Split the 'between' part into lines
    between_lines = between.split('\n')
    # Find the first non-empty line in the 'between' part and its indentation
    first_non_empty_line = next((line for line in between_lines if line.strip()), "")
    first_line_indent = calculate_indentation(first_non_empty_line, indent_char, tab_indent=tab_indent)

    # Detect if the last line in 'before_lines' is a control structure that requires an indented block
    if last_non_empty_line.strip().endswith((':', '{', 'def', 'class', 'if', 'elif', 'else', 'try', 'except', 'finally', 'with', 'for', 'while')):
        # Increase indent level for control structures
        if indent_char == '\t':
            last_before_line_indent += tab_indent  # For tabs, increase by 4 (equivalent spaces)
        else:
            last_before_line_indent += len(indent_char)  # For spaces, increase by the length of indent_char

    # If the last line in 'before_lines' is a closure of an indented code block, decrease the indent level
    if any(last_non_empty_line.strip().startswith(keyword) for keyword in ['return', 'break', 'continue', 'pass', '}', 'raise', 'yield']):
        if indent_char == '\t':
            last_before_line_indent -= tab_indent  # For tabs, decrease by 4 (equivalent spaces)
        else:
            last_before_line_indent -= len(indent_char)  # For spaces, decrease by the length of indent_char

    # If the 'between' part starts with a keyword that is outside the previous indentation, 
    # then change the indent level to align with the corresponding control structure
    if any(first_non_empty_line.startswith(keyword) for keyword in ['elif ', 'else:', 'except ', 'finally:']):
        # Find the counterpart keywords for the first keyword in 'between'
        counterpart_keywords = next((counterparts[keyword] for keyword in counterparts if first_non_empty_line.startswith(keyword)), None)
        if counterpart_keywords:
            # Find the previous line that starts with any of the counterpart keywords
            previous_non_empty_line = next((line for line in reversed(before_lines) if any(line.strip().startswith(keyword) for keyword in counterpart_keywords)), "")
            previous_line_indent = calculate_indentation(previous_non_empty_line, indent_char, tab_indent=tab_indent)
            # Align with the nearest line start with the counterpart
            last_before_line_indent = previous_line_indent
            
    # Calculate the base indentation adjustment for the 'between' part
    indentation_adjustment = last_before_line_indent - first_line_indent
    # Adjust the indentation of the 'between' part
    adjusted_between_lines = []
    for line in between_lines:
        if line.strip():
            # Calculate the new indentation
            current_indentation = calculate_indentation(line, indent_char, tab_indent=tab_indent)
            adjusted_indentation = current_indentation + indentation_adjustment
            adjusted_line = ' ' * max(adjusted_indentation, 0) + line.lstrip()
            # If the original code uses tabs for indentation, replace the spaces in the adjusted line with tabs
            if indent_char == '\t':
                adjusted_line = adjusted_line.replace(' ' * tab_indent, '\t')
            adjusted_between_lines.append(adjusted_line)
        else:
            adjusted_between_lines.append(line)  # Preserve empty lines as they are

    return '\n'.join(adjusted_between_lines)

def is_code_line(line, program_type, in_block_comment=False, store_comment=True):
    """
    Determine if a line is code, considering Python syntax and block comments.
    
    Parameters:
    line (str): The line to be checked.
    program_type (str): The programming language of the file.
    in_block_comment (bool): Whether the current line is within a block comment.
    store_comment (bool): Whether the code comment should be considered as code.
    
    Returns:
    triple: (bool, bool, bool) indicating if the line is considered , the current block comment state, and next line's block comment state.
    """
    line = line.strip()
    next_in_block_comment = in_block_comment

    # Common patterns
    single_line_comment = {'Python': '#', 'Java': '//'}
    block_comment_start = {'Python': ("'''", '"""'), 'Java': ('/*',)}
    block_comment_end = {'Python': ("'''", '"""'), 'Java': ('*/',)}

    # Check for block comment toggles in Python and Java
    if program_type == "Python":
        for q in block_comment_start['Python']:
            if q in line:
                count = line.count(q)
                if count % 2 != 0:
                    next_in_block_comment = not in_block_comment
                return store_comment, True, next_in_block_comment

    if in_block_comment:
        if any(end in line for end in block_comment_end[program_type]):
            next_in_block_comment = False
            return store_comment, True, False
        return store_comment, True, True

    # Checking for the start of a block comment in Java
    if any(start in line for start in block_comment_start[program_type]):
        if any(end in line for end in block_comment_end[program_type]):
            return store_comment, True, False
        next_in_block_comment = True
        return store_comment, True, next_in_block_comment

    # Specific code recognition logic for Python and Java
    if program_type == "Python":
        python_code_pattern = re.compile(
            r'^[\s]*(def |class |import |from |if |elif |else |try |except |'
            r'for |while |with |return |raise |assert |@|pass|'
            r'\w+\s*(=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|\*\*=|//=)\s*|'
            r'\w+\s*\(|\w+\[.*\]|\w+\]).*|'
            r'\w+\.\w+\(|'                            # Method calls
            r'(\[\w+\] \* \w+)|'                      # List operations like multiplication
            r'(\w+\s*\+=\s*\[.*\])|'                  # Compound assignment with list
            r'(\w+\s*\.\w+\(.*\))|'                   # More general method calls with arguments
            r'.*\w+\(.*\).*|'                         # Lines with function/method calls with arguments
            r'^[\s]*[\[\]\{\}]+[\s]*$|'               # Isolated or nested brackets/braces as the sole content
            r'[\w,\s]+\s*=\s*[\w.]+.*|'               # Variable assignments
            r'^[\s]*\],?$|'                           # Lines ending a list or dictionary, possibly with a comma
            r'^[\s]*\([^\)]*\),?$|'                   # Lines that likely represent tuple elements in a list
            r'^[\s]*\[[^\]]*\],?$|'                   # Lines that likely represent list elements in a list
            r'^[\s]*\{[^\}]*\},?$|'                   # Lines that likely represent dict entries in a list or dictionary
            r'\(\w+, \w+, \d+\)|'                     # Matches strings like "(equalities, eqThreshold, 0)"
            r'[\w,\s]+\s*,\s*(#.*|)$',                # Multi-line variable declarations ending with a comma, potentially followed by an inline comment
            re.DOTALL)                                # Match across multiple lines
        is_code = bool(python_code_pattern.match(line))
    elif program_type == "Java":
        java_code_pattern = re.compile(
            r'^[\s]*(import .*;|'                       # Add this line to match import statements
            r'public |private |protected |class |interface |@|enum |'
            r'void |static |final |if |for |while |switch |try |catch |'
            r'synchronized |new |return |assert |throw |extends |implements |'
            r'\w+\[\w*\]\s*=\s*[^;]*;|'                 # Array assignments
            r'\w+\s*=\s*[^;]*;|'                        # Simple assignments
            r'\w+\+\+;|\w+--;|'                         # Increment and decrement operations
            r'\w+<[^>]+>\([^)]*\)\s*{{.*?}}|'           # Matches complex generic definitions with anonymous classes
            r'\w+\(\)|'                                 # Constructor and method calls with no args
            r'\w+\.\w+\(|'                              # Method invocations like System.out.println
            r'\w+\s*\([^)]*\)|'                         # Method calls with args, including nested calls
            r'.*\);|'                                   # Method invocations ending with );
            r'\.\w+\s*->\s*|'                           # Lambda expressions
            r'.*::.*|'                                  # Method references
            r'\{|\}|'                                   # Individual braces
            r'//.*|'                                    # Single line comments
            r'".*"|'                                    # String literals
            r'\w+\s*<[^>]+>\s*[^;]*;|'                  # Generic types with assignments
            r'\.\w+\([^)]*\)|'                          # Method invocations on objects
            r'.*[^;{}]\s+\{.*|'                         # Lines that contain an open brace that aren't just a brace
            r'[\w\[\]]+\s*=\s*[^;]*;|'                  # Simple assignments
            r'[\w\[\]]+\s*/=\s*[^;]*;|'                 # Division assignments
            r'[\w\s]+[\w\[\]]+\s*=\s*[^;]*;|'           # Variable declarations and assignments
            r'\w+\s+\w+\s*;).*', re.DOTALL)             # General statements and variable declarations
        is_code = bool(java_code_pattern.match(line))

    if line.startswith(single_line_comment[program_type]):
        return store_comment, False, False

    return is_code, False, next_in_block_comment

def extract_code_sections(code, start_markers, end_markers):
    """
    Extract code sections from the provided text based on lists of start and end markers.
    
    Parameters:
    code (str): The text containing code sections.
    start_markers (list[str]): The markers indicating the start of a code section.
    end_markers (list[str]): The markers indicating the end of a code section.
    
    Returns:
    tuple: A tuple containing a list of extracted code sections and the index after the last code section.
    """
    sections = []
    index = 0
    last_section_end = 0  # Initialize the variable to track the end of the last section
    while index < len(code):
        section_found = False
        for start_marker in start_markers:
            start_idx = code.find(start_marker, index)
            if start_idx != -1:
                start_idx += len(start_marker)  # Adjust to get content just after the start_marker
                for end_marker in end_markers:
                    end_idx = code.find(end_marker, start_idx)
                    if end_idx != -1:
                        # Extract the code section
                        sections.append(code[start_idx:end_idx].rstrip())  # Use rstrip to remove trailing whitespace/newlines
                        index = last_section_end = end_idx + len(end_marker)  # Update index to continue after this section
                        section_found = True
                        break
                if section_found:
                    break
        if not section_found:
            break  # Exit loop if no more sections are found
    return sections, last_section_end  # Return both the sections and the last section end index

def extract_markdown_code_block(code, program_type):
    """
    Extracts the first code block for the specified programming language from a markdown formatted text.
    
    Parameters:
    code (str): The markdown containing the code blocks.
    program_type (str): The programming language of the code block to extract.
    
    Returns:
    str: The extracted code block content or None if no block is found.
    """
    # Define pairs of start and end markers
    markers = [
        (f"```{program_type.lower()}", "```"),
        (f"// --BEGIN MISSING CODE--" if program_type == "Java" else "# --BEGIN MISSING CODE--",
         f"// --END MISSING CODE--" if program_type == "Java" else "# --END MISSING CODE--")
    ]

    for start_marker, end_marker in markers:
        # Search for code block
        pattern = re.compile(rf"{re.escape(start_marker)}(.*?){re.escape(end_marker)}", re.DOTALL)
        matches = pattern.findall(code)
        if matches:
            # Return the first match's content without stripping leading/trailing whitespace
            return matches[0]

    return None  # Return None if no block is found

def clean_code(code, program_type, check_brackets=False, keep_comments=False, return_first_code_snippet=True):
    """
    Clean the provided code by removing non-code syntax, language dividers, comments.
    (optional)Ensuring that the code structure is complete with all brackets closed.
    
    Parameters:
    code (str): The code to be cleaned.
    program_type (str): The programming language of the code.
    
    Returns:
    str: The cleaned code.
    """
    if return_first_code_snippet:
        # Attempt to extract code from markdown block first
        extracted_code = extract_markdown_code_block(code, program_type)
        if extracted_code is not None:
            return extracted_code
    
    code = re.sub(r'^```.*$', '', code, flags=re.MULTILINE)  # Remove Markdown code fences if present
    lines = code.split('\n')
    processed_lines = []
    structure_depth = {'(': 0, '[': 0, '{': 0}
    bracket_pairs = {')': '(', ']': '[', '}': '{'}
    in_block_comment = False

    for line in lines:
        # stripped_line = line.strip()
        is_code, in_block_comment, next_in_block_comment = is_code_line(line, program_type, in_block_comment=in_block_comment, store_comment=keep_comments)
        # Skip lines that are comments or within block comments
        if in_block_comment or not is_code:
            continue
    
        in_block_comment = next_in_block_comment

        processed_lines.append(line)

        # Update structure depth based on the presence of brackets
        for bracket in '([{':
            structure_depth[bracket] += line.count(bracket)
        for bracket in ')]}':
            structure_depth[bracket_pairs[bracket]] -= line.count(bracket)
    if check_brackets:
        # Check if all structures are closed, and potentially append closing brackets if not.
        # This simplistic example does not automatically append closing brackets but warns about unclosed structures.
        unclosed_brackets = [bracket for bracket, depth in structure_depth.items() if depth > 0]
        if unclosed_brackets:
            print(f"Warning: Unclosed brackets detected {unclosed_brackets}. Consider reviewing the code for completeness.")

    return '\n'.join(processed_lines).rstrip()

def is_fuzzy_similar(a, b, threshold=0.98):
    """
    Determine if two strings are similar based on a threshold using fuzzy matching.
    Ignores differences in spaces between characters.

    Args:
    a (str): First string.
    b (str): Second string.
    threshold (float): Similarity threshold.

    Returns:
    bool: True if the strings are similar, False otherwise.
    """
    # Remove spaces from both strings for comparison
    a_no_spaces = ''.join(a.split())
    b_no_spaces = ''.join(b.split())

    ratio = SequenceMatcher(None, a_no_spaces, b_no_spaces).ratio()
    return ratio > threshold

def normalize_line(line):
    """
    Normalize a line by stripping leading/trailing spaces and reducing all internal spaces to single spaces.
    """
    return ' '.join(line.split())

def find_longest_match_with_comments(matches, between_lines, program_type, is_start=True):
    if not matches:
        return 0 if is_start else len(between_lines)
    
    longest_sequence_start = matches[0]
    longest_sequence_end = matches[0]
    current_sequence_start = matches[0]
    current_sequence_end = matches[0]
    max_sequence_length = 1
    current_sequence_length = 1

    in_block_comment = False
    for i in range(1, len(matches)):
        is_continuous = matches[i] == current_sequence_end + 1
        # Check if the next line is code to determine continuity
        # Sine this step is after the clean step, if not code then it is a comment
        next_is_code, in_block_comment, next_in_block_comment = is_code_line(between_lines[current_sequence_end + 1], program_type, in_block_comment=in_block_comment, store_comment=False)
        is_comment_continuity = matches[i] == current_sequence_end + 2 and not next_is_code
        in_block_comment = next_in_block_comment

        if is_continuous or is_comment_continuity:
            current_sequence_end = matches[i]
            current_sequence_length += 1
            if current_sequence_length > max_sequence_length:
                max_sequence_length = current_sequence_length
                longest_sequence_start = current_sequence_start
                longest_sequence_end = current_sequence_end
        else:
            current_sequence_start = matches[i]
            current_sequence_end = matches[i]
            current_sequence_length = 1

    # Decide what to return based on whether we're looking for a start or end index
    if is_start:
        return longest_sequence_end
    else:
        return longest_sequence_start

# Introduce the ignore_lines
def find_first_continuous_match(between_lines, comparison_lines, reverse=False, min_matches=2):
    if reverse:
        comparison_lines = list(reversed(comparison_lines))

    last_match_index = None  # Track the last match index in between_lines
    continuous_matches = []  # Track indexes of continuous matches in between_lines

    # Define a set of lines to ignore
    ignore_lines = set(['{', '}', '(', ')', '[', ']',''])
    # ignore_lines = set()


    # Iterate through comparison_lines to find matches in between_lines
    for comp_line in comparison_lines:
        for i, b_line in enumerate(between_lines):
            normalized_comp_line = normalize_line(comp_line)
            normalized_b_line = normalize_line(b_line)

            # Skip if the line is in the ignore set
            if normalized_comp_line in ignore_lines or normalized_b_line in ignore_lines or not normalized_b_line.strip():
                continue

            if normalized_comp_line == normalized_b_line:
                # Check if this match is continuous with the previous match
                if last_match_index is None or abs(i - last_match_index) == 1:
                    continuous_matches.append(i)
                    last_match_index = i
                else:
                    # If a new match is not continuous, reset if not met min_matches yet
                    if len(continuous_matches) < min_matches:
                        continuous_matches = [i]
                        last_match_index = i
                break  # Proceed to the next comparison_line after finding a match
    # Check if we've found a sufficient continuous match
    if len(continuous_matches) >= min_matches:
        if reverse:
            # For optimal_start, we return the index following the continuous sequence
            return continuous_matches[0] + 1
        else:
            # For optimal_end, return the start index of the continuous sequence
            return continuous_matches[0]
    elif continuous_matches:
        # If no continuous match meets the criteria, default to the first match
        if reverse:
            # For optimal_start, we return the index following the continuous sequence
            return continuous_matches[0] + 1
        else:
            # For optimal_end, return the start index of the continuous sequence
            return continuous_matches[0]
    else:
        # If no continuous match meets the criteria, default to start or end
        return 0 if reverse else len(between_lines)

def check_trimming_ordering(optimal_start, optimal_end, between_lines, trim_choice='tail'):
    # Ensure the selected optimal_start and optimal_end satisfy the logical ordering
    if optimal_start <= optimal_end:
        return optimal_start, optimal_end
    else:
        if trim_choice == "tail":
            optimal_start, optimal_end = 0, optimal_end
        elif trim_choice == "head":
            optimal_start, optimal_end = optimal_start, len(between_lines)
        else:
            optimal_start, optimal_end = 0, len(between_lines)  # Adjust as needed for non-overlapping or invalid ranges

    # If the two return values are equal, return the entire range
    if optimal_start == optimal_end:
        return 0, len(between_lines)

    return optimal_start, optimal_end        

def evaluate_optimal_indexes(before_lines, between_lines, after_lines, min_matches=2):
    optimal_start = find_first_continuous_match(between_lines, before_lines, reverse=True, min_matches=min_matches)
    optimal_end = find_first_continuous_match(between_lines, after_lines, reverse=False, min_matches=min_matches)

    # Ensure the selected optimal_start and optimal_end satisfy logical ordering and bounds
    optimal_start = max(0, optimal_start)
    optimal_end = min(len(between_lines), optimal_end)

    # Default values if no matches found
    if optimal_start is None: optimal_start = 0
    if optimal_end is None: optimal_end = len(between_lines)

    # Ensure the selected optimal_start and optimal_end satisfy the logical ordering
    return check_trimming_ordering(optimal_start, optimal_end, between_lines, trim_choice='tail')

def remove_1st_duplicates(before_lines_stripped, between_lines, optimal_start, optimal_end, program_type):
    # Find the last line of code (non-comment) in before_lines_stripped
    last_non_comment_line_in_before = None
    in_block_comment = False
    for line in reversed(before_lines_stripped):
        # Check if the line is code
        is_code, in_block_comment, next_in_block_comment = is_code_line(line, program_type, in_block_comment=in_block_comment, store_comment=False)
        in_block_comment = next_in_block_comment
        if is_code:
        # if not is_code_comment(line):
            last_non_comment_line_in_before = line
            break
    # Check if optimal_start duplicates the last line of code in before_lines_stripped
    if last_non_comment_line_in_before is not None and optimal_start < len(between_lines):
        if normalize_line(between_lines[optimal_start]) == normalize_line(last_non_comment_line_in_before):
            # Increment optimal_start if it does not cause an out-of-index error
            if optimal_start + 1 < len(between_lines) and optimal_start + 1 <= optimal_end:
                optimal_start += 1
    return optimal_start

def remove_n_duplicates(before_lines_stripped, between_lines, optimal_start, optimal_end, program_type, n=1):
    # Reverse iterate through before_lines_stripped to find the last n lines of code (non-comment)
    code_lines_to_compare = []
    in_block_comment = False
    for line in reversed(before_lines_stripped):
        # Check if the line is code
        is_code, in_block_comment, next_in_block_comment = is_code_line(line, program_type, in_block_comment=in_block_comment, store_comment=False)
        in_block_comment = next_in_block_comment
        if is_code:
        # if not is_code_comment(line):
            code_lines_to_compare.append(normalize_line(line))
            if len(code_lines_to_compare) == n:
                break
    
    # Reverse the list to match the original order in before_lines
    code_lines_to_compare.reverse()
    
    # Check if the beginning of between_lines matches the last n lines of code in before_lines_stripped
    matching_lines_count = 0
    for i, line in enumerate(code_lines_to_compare):
        # Ensure we do not exceed the bounds of between_lines when comparing
        if optimal_start + i < len(between_lines) and optimal_start + i <= optimal_end:
            if normalize_line(between_lines[optimal_start + i]) == line:
                matching_lines_count += 1
            else:
                break  # Stop checking further if any line does not match
    
    # Adjust optimal_start based on the number of matching lines found
    if matching_lines_count == n:
        optimal_start += matching_lines_count  # Increment to skip these matching lines
    
    # Ensure optimal_start does not exceed bounds after adjustment
    optimal_start = min(optimal_start, len(between_lines) if optimal_end >= len(between_lines) else optimal_end + 1)
    
    return optimal_start

def extract_non_comment_code_lines_with_indices(lines, program_type):
    # Initialize an empty list to hold the result
    result = []
    in_block_comment = False
    # Iterate through lines with indices
    for idx, line in enumerate(lines):
        # Check if the line is code
        is_code, in_block_comment, next_in_block_comment = is_code_line(line, program_type, in_block_comment=in_block_comment, store_comment=False)
        in_block_comment = next_in_block_comment
        if is_code:
        # if not is_code_comment(line):
            # Normalize the line and append the tuple (index, line) to the result
            result.append((idx, normalize_line(line)))
    
    # Return the list of non-comment lines with their indices
    return result


# Function to compare and adjust indices based on matches
def compare_and_adjust(start_or_end, lines_to_compare, between_filtered, is_start=True):
    match_count = 0
    for (comp_idx, comp_line) in lines_to_compare:
        for (between_idx, between_line) in between_filtered:
            if is_fuzzy_similar(comp_line, between_line):
                match_count += 1
                if match_count == 1 and is_start:
                    # For optimal_start, adjust to the next line after the last match
                    return between_idx + 1
                elif not is_start:
                    # For optimal_end, adjust based on the first match
                    return between_idx
                break  # Proceed to the next comparison line after a match
        else:
            # Reset if a comparison line doesn't match
            match_count = 0
    return start_or_end  # Return original value if no adjustments needed

def adjust_optimal_indexes(before_lines_stripped, between_lines, after_lines, optimal_start, optimal_end, program_type, n=2):
    # Extract non-comment code lines and their indices from all line groups
    before_lines_filtered = extract_non_comment_code_lines_with_indices(before_lines_stripped, program_type)[-n:]
    between_lines_filtered = extract_non_comment_code_lines_with_indices(between_lines[optimal_start: optimal_end], program_type)
    after_lines_filtered = extract_non_comment_code_lines_with_indices(after_lines, program_type)[:n]
    # Store the original values
    original_start = optimal_start
    original_end = optimal_end
    # Adjust optimal_start and optimal_end based on matching lines
    optimal_start = compare_and_adjust(optimal_start, before_lines_filtered, between_lines_filtered, is_start=True)
    optimal_end = compare_and_adjust(optimal_end, after_lines_filtered, between_lines_filtered, is_start=False)
    # Check if optimal_start equals optimal_end and if the original optimal_start is less than the original optimal_end
    if optimal_start >= optimal_end and original_start <= original_end:
        # If true, set optimal_start and optimal_end to their original values
        optimal_start = original_start
        optimal_end = original_end
    # Ensure optimal_start and optimal_end do not exceed bounds
    optimal_start = max(0, optimal_start)
    optimal_end = min(optimal_end, len(between_lines))

    return optimal_start, optimal_end

def evaluate_optimal_matches(before_lines_stripped, between_lines, after_lines_stripped, program_type, match_method = 'first_continuous'):
    if match_method == 'first_continuous':
        optimal_start, optimal_end = evaluate_optimal_indexes(before_lines_stripped, between_lines, after_lines_stripped, min_matches=1)
    elif match_method == 'longest_continuous':
        # Gather potential matches
        start_matches = gather_matches(between_lines, before_lines_stripped, search_from_end=True)
        end_matches = gather_matches(between_lines, after_lines_stripped, search_from_end=False)
        # Find the longest continuous block considering comments
        optimal_start = find_longest_match_with_comments(start_matches, between_lines, program_type, is_start=True)
        optimal_end = find_longest_match_with_comments(end_matches, between_lines, program_type, is_start=False)
    elif match_method == 'keep_all':
        optimal_start = 0
        optimal_end = len(between_lines)
    else:
        raise ValueError("Invalid match_method. Use 'first_continuous' or 'longest_continuous' or 'keep_all'")
    # Adjust optimal start and end
    optimal_start, optimal_end = adjust_optimal_indexes(before_lines_stripped, between_lines, after_lines_stripped, optimal_start, optimal_end, program_type, n=2)
    
    # Attempt to find the second-best pair if the primary condition is not met
    if optimal_start is not None and optimal_end is not None:
        if optimal_start < optimal_end:
            return optimal_start, optimal_end
        elif optimal_start == optimal_end:
            return 0, len(between_lines)
        else:
            return check_trimming_ordering(optimal_start, optimal_end, between_lines, trim_choice='tail')
            # return None, None  # Placeholder for sub-optimal logic

    return 0, len(between_lines)  # Default values if no suitable matches are found

def gather_matches(target_lines, search_lines, search_from_end=False):
    """
    Gathers all potential matches between target_lines and search_lines.
    If search_from_end is True, matches are searched from the end of search_lines.
    """
    matches = []
    # search_lines_normalized = [normalize_line(line) for line in search_lines]
    search_lines_normalized = [normalize_line(line) for line in (reversed(search_lines) if search_from_end else search_lines)]
    in_block_comment = False

    for i, line in enumerate(target_lines):
        if line.strip().startswith('#') or '"""' in line or "'''" in line:
            if '"""' in line or "'''" in line:
                in_block_comment = not in_block_comment
            if in_block_comment or line.strip().startswith('#'):
                continue

        line_normalized = normalize_line(line)
        for other_line in search_lines_normalized:
            if is_fuzzy_similar(line_normalized, other_line):
                matches.append(i)
                break  # Stop searching once a match is found for this line

    return matches

def find_sub_optimal_start_end(between_lines, start_matches, end_matches):
    #Initialize optimal indexes
    optimal_start, optimal_end = 0, len(between_lines)
    # Evaluate matches to find the optimal indexes
    for start in start_matches:
        for end in end_matches:
            if start <= end and (end - start > optimal_end - optimal_start or start < optimal_start):
                optimal_start, optimal_end = start, end
                break  # Found an optimal match, no need to continue for this start
    return optimal_start, optimal_end

def find_continuous_similarity(between_lines, before_lines_stripped, after_lines_stripped, program_type):
    """
    Find and select the optimal start and end index for matches between 'between_lines' and
    'before_lines_stripped'/'after_lines_stripped', adhering to specified constraints.
    """
    optimal_start, optimal_end = evaluate_optimal_matches(before_lines_stripped, between_lines, after_lines_stripped, program_type)
    # Ensure the optimal indexes meet the constraints
    # if optimal_start == None or optimal_end == None or optimal_start == optimal_end:
    # Ensure the optimal indexes meet the constraints - not None, Equal, or no code lines
    if (optimal_start == None or optimal_end == None or optimal_start == optimal_end or
        not extract_non_comment_code_lines_with_indices(between_lines[optimal_start:optimal_end], program_type)):
        optimal_start, optimal_end = 0, len(between_lines)  # Adjust based on specific logic if needed

    return optimal_start, optimal_end

def find_continuous_blocks_similarity(between_lines, other_lines, from_start=True):
    """
    Find the continuous block of lines in 'between_lines' that are similar to lines in 'other_lines'.
    """
    other_lines_normalized = [normalize_line(line) for line in other_lines if line.strip()]

    if from_start:
        last_match_end = -1  # Initialize to an invalid value
        continuous_block_length = 0  # Length of the continuous block
        for i, line in enumerate(between_lines):
            line_normalized = normalize_line(line)
            if any(is_fuzzy_similar(line_normalized, other_line) for other_line in other_lines_normalized):
                continuous_block_length += 1
                # If it's a match, mark the end of this matching block
                if continuous_block_length > 1:
                    last_match_end = i + 1
            elif continuous_block_length > 0:
                # If it was a match but the current line isn't, stop the search
                break
        # If no valid match was found (two or more lines), start_index should be 0
        return last_match_end if last_match_end != -1 else 0

    else:
        earliest_match_start = len(between_lines)  # Initialize to the end of the list
        continuous_block_length = 0  # Length of the continuous block
        for i, line in enumerate(reversed(between_lines)):
            line_normalized = normalize_line(line)
            if any(is_fuzzy_similar(line_normalized, other_line) for other_line in other_lines_normalized):
                continuous_block_length += 1
                # If it's a match, update the earliest_match_start
                if continuous_block_length > 1:
                    earliest_match_start = min(earliest_match_start, len(between_lines) - i - 1)
            elif continuous_block_length > 0:
                # If a match was found but the current line isn't a match, stop the search
                break
        # If no valid match was found (two or more lines), end_index should be the length of between_lines
        return earliest_match_start if continuous_block_length > 1 else len(between_lines)


def get_indentation(line):
    """
    Returns the indentation of the given line.
    """
    return len(line) - len(line.lstrip())

def trim_similar_edges(between, before_lines, after_lines, program_type):
    # Usage in trim_similar_edges function
    between_lines = between.split('\n')
    before_lines_stripped = [line.strip() for line in before_lines]
    after_lines_stripped = [line.strip() for line in after_lines]
    start_index, end_index = find_continuous_similarity(between_lines, before_lines_stripped, after_lines_stripped, program_type)
    # Trim the similar edges
    between_cleaned = between_lines[start_index:end_index]
    return '\n'.join(between_cleaned)
