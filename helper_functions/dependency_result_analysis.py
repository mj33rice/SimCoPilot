import pandas as pd
import re
from collections import Counter

def define_patterns():
    # Define the regex patterns
    horizon_pattern = (
        r"(?P<def_type>Variable|Function|Class|Interface|Library|Global_Variable) '.*?' used at line \d+ "
        r"(?:is (?:imported|defined) at line \d+ and has a (?P<range_type>Short-Range|Medium-Range|Long-Range) dependency\.|"
        r"is part of a (?P<comp_type>List_Comprehension|Generator_Expressions|Lambda_Expressions|Loop) defined at line \d+ and has a (?P<comp_range_type>Short-Range|Medium-Range|Long-Range) dependency\.)"
    )
    reason_pattern = r"'(If Condition|If Body|Elif Condition|Elif Body|Else Reasoning|Pattern-Based Completion|Define Stop Criteria|Loop Body|Generator_Expressions|Super_Call|List_Comprehension|Lambda_Expressions|Stream_Operations)' detected at line \d+\."
    return horizon_pattern, reason_pattern

def analyze_and_get_frequencies(input_data, column_name, pattern):
    # Check if input_data is a DataFrame or a single row (list of strings)
    if isinstance(input_data, pd.DataFrame):
        task_categories = []
        for index, row in input_data.iterrows():
            cell_content = row[column_name]
            if pd.isna(cell_content):
                task_categories.append({})
                continue
            cell_content = str(cell_content)
            task_categories.append(perform_frequency_analysis(cell_content, pattern))
        return task_categories
    elif isinstance(input_data, list) or isinstance(input_data, str):
        return perform_frequency_analysis(input_data, pattern)
    else:
        raise ValueError("Input data must be a pandas DataFrame or a list of strings.")

def perform_frequency_analysis(input_content, pattern):
    # Ensure input_content is a string
    input_string = "\n".join(input_content) if isinstance(input_content, list) else input_content

    # Extract the information
    extracted_info = re.findall(pattern, input_string, flags=re.DOTALL)

    # Initialize an empty Counter for frequency analysis
    freq_analysis = Counter()

    # Determine if we are processing horizon categories or reason categories
    is_horizon_category = 'comp_type' in pattern or 'range_type' in pattern

    for match in extracted_info:
        if is_horizon_category:
            # For horizon categories, handle based on the presence of comp_type or range_type
            if match[2]:  # If comp_type is present (list comprehension or loop)
                category = f"{match[0]} {match[2]} {match[3]}"  # Use comp_type and comp_range_type
            else:
                category = f"{match[0]} {match[1]}"  # Use def_type and range_type
        else:
            # For reason categories, the match is assumed to be a tuple with the whole category as the first element
            category = match if isinstance(match, str) else match[0]

        freq_analysis[category] += 1

    # Convert the Counter object to a dictionary
    return dict(freq_analysis)



def get_task_horizon_reason_labels(df):
    horizon_pattern, reason_pattern = define_patterns()
    # Perform the analysis and get frequencies
    df['task_horizon_categories'] = analyze_and_get_frequencies(df, 'horizon_categories_output', horizon_pattern)
    df['task_reason_categories'] = analyze_and_get_frequencies(df, 'reason_categories_output', reason_pattern)
    return df


# # Set display options
# pd.set_option('display.max_columns', None)  # Ensures all columns are displayed
# pd.set_option('display.width', 2)        # Sets display width to avoid wrapping of columns
# pd.set_option('display.max_colwidth', None) # Displays full content of each column

# # Path to your CSV file
# csv_file_path = '../Analysis_Results/analysis_results_2.csv'
# # Load the CSV file into a pandas DataFrame
# df = pd.read_csv(csv_file_path)
# # Display the DataFrame
# print(df)
# df = get_task_horizon_reason_labels(df)
# # Save the updated DataFrame to a new CSV file
# output_file_path = '../Analysis_Results/updated_analysis_results.csv'
# df.to_csv(output_file_path, index=False)
# # Display a message to confirm the file has been saved
# print(f"DataFrame successfully saved to {output_file_path}")
# import pdb;pdb.set_trace()

# # Example usage with a single row
# reason_categories_output = [
#     "'If-else Reasoning' detected at line 43.",
#     "'If-else Reasoning' detected at line 45.",
#     "'If-else Reasoning' detected at line 47."
# ]
# horizon_categories_output = ["Variable 'ltThreshold' used at line 41 is local defined at line 34 and has a Medium-Range dependency.", "Variable 'newVars' used at line 41 is local defined at line 35 and has a Medium-Range dependency.", "Variable 'numRows' used at line 42 is local defined at line 30 and has a Medium-Range dependency.", "Variable 'ltThreshold' used at line 42 is local defined at line 34 and has a Medium-Range dependency.", "Variable 'eqThreshold' used at line 43 is local defined at line 34 and has a Medium-Range dependency.", "Variable 'eqThreshold' used at line 44 is local defined at line 34 and has a Medium-Range dependency.", "Variable 'numRows' used at line 44 is local defined at line 30 and has a Medium-Range dependency.", "Variable 'maximization' used at line 45 is local defined at line 34 and has a Medium-Range dependency.", "Variable 'x' used at line 46 is part of a list_comp.", "Variable 'newVars' used at line 47 is local defined at line 35 and has a Medium-Range dependency.", "Variable 'cost' used at line 48 is local defined at line 34 and has a Medium-Range dependency.", "Variable 'equalities' used at line 48 is local defined at line 34 and has a Medium-Range dependency.", "Variable 'eqThreshold' used at line 48 is local defined at line 34 and has a Medium-Range dependency."]
# horizon_pattern, reason_pattern = define_patterns()
# reason_freq_analysis = analyze_and_get_frequencies(reason_categories_output, None, reason_pattern)
# horizon_freq_analysis = analyze_and_get_frequencies(horizon_categories_output, None, horizon_pattern)

# print(reason_freq_analysis)
# print(horizon_freq_analysis)
