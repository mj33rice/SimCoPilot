from termcolor import colored, cprint

def add_line_numbers_and_color(lines, start=1, text_color=None):
    """
    Adds line numbers and optional color to each line of the given code.

    :param lines: List of code lines to be processed.
    :param start: Starting line number.
    :param text_color: Optional color for the text.
    :return: String representation of the code with line numbers and optional color.
    """
    colored_lines = []
    for i, line in enumerate(lines, start):
        line_number = colored(f'{i:4}: ', 'magenta')
        colored_line = line_number + colored(line, text_color) if text_color else line_number + line
        colored_lines.append(colored_line)
    return '\n'.join(colored_lines)

def replace_tabs_with_spaces(code, spaces_per_tab=4):
    """
    Replaces all tab characters in the given code with a specified number of spaces.

    :param code: The code to process.
    :param spaces_per_tab: The number of spaces to replace each tab with.
    :return: The code with tabs replaced by spaces.
    """
    return code.replace('\t', ' ' * spaces_per_tab)

def split_code(source_code, start_line, end_line):
    """
    Splits the source code into three parts: before, between, and after the specified line numbers.
    """
    source_code = replace_tabs_with_spaces(source_code)
    lines = source_code.split('\n')
    return lines[:start_line - 1], lines[start_line - 1:end_line], lines[end_line:]

def colored_line_numbers_and_text(before, between, after, start_line, end_line, text_color='cyan'):
    """
    Adds colored line numbers and text to the 'before', 'between', and 'after' parts of code.

    :param before: Lines of code before the target segment.
    :param between: Lines of code in the target segment.
    :param after: Lines of code after the target segment.
    :param start_line: The starting line number of the target segment (1-indexed).
    :param end_line: The ending line number of the target segment (1-indexed).
    :param text_color: The color for the text in the target segment.
    :return: Tuple containing colored representations of 'before', 'between', and 'after' code segments.
    """
    colored_before = add_line_numbers_and_color(before) if before else ''
    colored_between = add_line_numbers_and_color(between, start=start_line, text_color=text_color)
    colored_after = add_line_numbers_and_color(after, start=end_line + 1) if after else ''

    return colored_before, colored_between, colored_after

