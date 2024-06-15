import os
import re
import pandas as pd
from datetime import datetime

def overwrite_columns(sor_csv_file, tar_csv_file, columns_to_overwrite, keys, create_new_cols=True, cols_to_ignore=[]):
    # Load the CSV files
    df_a = pd.read_csv(sor_csv_file)
    df_b = pd.read_csv(tar_csv_file)

    # Check if all specified columns exist in df_a
    missing_columns = [col for col in columns_to_overwrite if col not in df_a.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} are missing in the source file.")
    
    #Check if the keys match in both dataframes
    if not df_a[keys].equals(df_b[keys]):
        raise ValueError("Values in keys do not match between the two dataframes.")
    ########################################################
    # # Find the rows where the keys do not match
    # mismatched_rows = df_a[keys] != df_b[keys]

    # # Print the indices of the mismatched rows
    # print("Indices of mismatched rows:", mismatched_rows[mismatched_rows].index)

    # # Print the total number of mismatched rows
    # print("Total number of mismatched rows:", mismatched_rows.sum())

    # # Remove the mismatched rows from df_a
    # df_a = df_a[~mismatched_rows]

    # # Remove the mismatched rows from df_b
    # df_b = df_b[~mismatched_rows]
    ########################################################
    
    # Check if there's only one matching row in df_a for each row in df_b
    if df_a[keys].duplicated().any():
        raise ValueError("There are multiple matching rows in the source file for a single row in the target file.")
    
    # Merging DataFrames on keys
    df_b = df_b.merge(df_a[keys + columns_to_overwrite], on=keys, how='inner', suffixes=('', '_from_a'))

    # If create_new_cols is True, add columns from df_a to df_b if they don't exist in df_b
    if create_new_cols:
        for col in columns_to_overwrite:
            if col not in df_b.columns:
                df_b[col] = df_a[col]

    # Overwriting the columns in df_b with those from df_a where matches were found
    for col in columns_to_overwrite:
        if col + '_from_a' in df_b.columns:
            df_b[col] = df_b[col + '_from_a']
            df_b.drop(col + '_from_a', axis=1, inplace=True)

    # Remove the columns to ignore from df_b
    df_b = df_b.drop(columns=cols_to_ignore, errors='ignore')
    df_a = df_a.drop(columns=cols_to_ignore, errors='ignore')

    # Reorder the columns of df_b to match the order of df_a
    df_b = df_b[df_a.columns]

    return df_b

def rename_file(tar_csv_file):
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
    res_csv_file = os.path.join('merged', new_base_name + '.csv')

    return res_csv_file

def overwrite_columns_in_all_files(sor_csv_file, tar_csv_folder, columns_to_overwrite, keys, create_new_cols=True):
    tar_csv_files_list = os.listdir(tar_csv_folder)

    for tar_csv_file in tar_csv_files_list:
        if not tar_csv_file.endswith('.csv'):
            continue

        tar_csv_file_path = os.path.join(tar_csv_folder, tar_csv_file)

        df_b = overwrite_columns(sor_csv_file, tar_csv_file_path, columns_to_overwrite, keys, create_new_cols)

        res_csv_file_path = os.path.join(tar_csv_folder, rename_file(tar_csv_file))
        df_b.to_csv(res_csv_file_path, index=False)
        print(f"Columns overwritten successfully where matches found, and file {tar_csv_file_path} updated.")

def process_horizon_categories_output(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Initialize an empty list to store the results
    results = []

    # Define the regular expression pattern
    pattern = r"(?P<type>\w+) '.*?' used at line (?P<usage_line>\d+) is defined at line (?P<def_line>\d+)"

    # Iterate over each row in the 'horizon_categories_output' column
    for row in df['horizon_categories_output']:
        # Use the re.findall function to find all matches in the row
        matches = re.findall(pattern, row)

        # Convert the matches to the required format and add them to the results list
        for match in matches:
            type, usage_line, def_line = match
            results.append((type, (int(usage_line), int(def_line))))

    return results

# Example usage:
sor_csv_file = '../Analysis_Results/Post_process/Labels_ref/gpt-3.5-turbo-0125_with_afterlines_SparseArrayTester_05_07_23_35.csv'
tar_csv_folder = '../Analysis_Results/Post_process/SparseArrayTester/'

keys = ['before','between','after']
columns_to_overwrite = ['start_line', 'end_line', 'code_task', 'reason_categories_output', 'horizon_categories_output', 'reason_freq_analysis', 'horizon_freq_analysis']
overwrite_columns_in_all_files(sor_csv_file, tar_csv_folder, columns_to_overwrite, keys, create_new_cols=True)
