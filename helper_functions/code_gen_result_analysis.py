import ast
import os
import re
import pandas as pd
import numpy as np

def bootstrap_resampling(pass_count, total_count, num_resamples=10000):
    # Check if total_count is zero
    if total_count == 0:
        # Return default values for performance and err_bar
        return 0, 0
    
    # Calculate model's performance
    performance = pass_count / total_count

    # Generate bootstrap resamples
    resamples = np.random.choice([0, 1], size=(num_resamples, total_count), p=[1-performance, performance])

    # Calculate pass count for each resample
    resample_pass_counts = resamples.sum(axis=1)

    # Calculate performance for each resample
    resample_performances = resample_pass_counts / total_count

    # Calculate average and 1.96 standard deviations of resample performances
    avg_performance = resample_performances.mean()
    std_dev_performance = resample_performances.std()

    return avg_performance, 1.96 * std_dev_performance

def cal_err_bar(pass_counts, total_counts, num_resamples=10000):
    percentages = []
    err_bars = []
    for pass_count, total_count in zip(pass_counts, total_counts):
        # Use bootstrap resampling to calculate average performance and error bar
        percentage, err_bar = bootstrap_resampling(pass_count, total_count, num_resamples)
        percentages.append(percentage)
        err_bars.append(err_bar)

    return pd.Series(percentages, index=total_counts.index), pd.Series(err_bars, index=total_counts.index)

def overwrite_first_word_in_dict_keys(d, label_mapping):
    new_dict = {}
    for key, value in d.items():
        words = key.split(' ')
        if words[0] in label_mapping:
            words[0] = label_mapping[words[0]]
        new_key = ' '.join(words)
        new_dict[new_key] = value
    return new_dict
class CodeAnalysis:
    def __init__(self, gen_code_pass_col='gen_code_pass_ratio', reason_analysis_col='reason_freq_analysis', horizon_analysis_col='horizon_freq_analysis', pass_ratio_label='3/3', labels_reasons=None, labels_horizons=None, weighted=False, use_max_range=True):
        # Parameters for column names
        self.gen_code_pass_col = gen_code_pass_col
        self.reason_analysis_col = reason_analysis_col
        self.horizon_analysis_col = horizon_analysis_col
        # self.pass_ratio_label = pass_ratio_label

        # Initialize the labels for different types of analyses
        # self.labels_reasons = labels_reasons if labels_reasons is not None else ['List_Comprehension', 'Lambda_Expressions', 'Generator_Expressions', 'If-else Reasoning', 'Stream_Operations', 'Define Stop Criteria', 'Super_Call']
        self.labels_reasons = labels_reasons if labels_reasons is not None else ['List_Comprehension', 'Lambda_Expressions', 'Generator_Expressions', \
                                                                                 'If Condition','If Body', 'Elif Condition', 'Elif Body', 'Else Reasoning', \
                                                                                 'Stream_Operations', 'Loop Body', 'Define Stop Criteria', 'Super_Call']
        self.labels_horizons = labels_horizons if labels_horizons is not None else ['Short-Range', 'Medium-Range', 'Long-Range', 'Variable', 'Global_Variable', 'Function', 'Class', 'Library', 'Interface']

        self.weighted = weighted  # Boolean flag to determine if a weighted average should be used
        self.use_max_range = use_max_range  # Boolean flag to determine if the maximum range should be used

        # Define the range labels
        self.range_labels = ['Short-Range', 'Medium-Range', 'Long-Range']

    @staticmethod
    def format_as_percentage(passed, total, report_err_bar=True):
        if total > 0:
            if report_err_bar:
                percentage, err_bar = bootstrap_resampling(passed, total)
                return "{:.2f}% ± {:.2f} ({}/{})".format(percentage*100, err_bar*100, passed, total)
            else:
                return "{:.2f}% ({}/{})".format((passed / total) * 100, passed, total)
        else:
            return "N/A"

    @staticmethod
    def parse_frequency_data(frequency_analysis):
        if isinstance(frequency_analysis, dict):
            return frequency_analysis
        try:
            return ast.literal_eval(frequency_analysis)
        except ValueError:
            # print(f"Failed to parse: {frequency_analysis}")
            return {}
        
    @staticmethod
    def get_max_range_label(keys):
        range_order = ['Short-Range', 'Medium-Range', 'Long-Range']
        max_label = None
        for key in keys:
            for label in key.split():
                if label in range_order and (max_label is None or range_order.index(label) > range_order.index(max_label)):
                    max_label = label
        return max_label

    def get_pass_ratio(self, df, label, analysis_col):
        if self.weighted:
            return self.weighted_average_pass_ratio(df, label, analysis_col)
        else:
            return self.boolean_pass_ratio(df, label, analysis_col)

    def boolean_pass_ratio(self, df, label, analysis_col):
        pass_count = 0
        applicable_cases = 0
        for index, row in df.iterrows():
            analysis_data = self.parse_frequency_data(row[analysis_col])
            label_present = False
            if analysis_col == self.horizon_analysis_col and self.use_max_range and label in self.range_labels:
                max_label = self.get_max_range_label(analysis_data.keys())
                label_present = label == max_label
            elif analysis_col == self.horizon_analysis_col:
                label_present = any(label in key.split() for key in analysis_data.keys())
            elif analysis_col == self.reason_analysis_col:
                label_present = label in analysis_data

            if label_present:
                applicable_cases += 1
                if eval(row[self.gen_code_pass_col]) == 1:
                    pass_count += 1
        return pass_count, applicable_cases

    def weighted_average_pass_ratio(self, df, label, analysis_col):
        weighted_pass = 0
        weighted_total = 0
        for index, row in df.iterrows():
            frequency_data = self.parse_frequency_data(row[analysis_col])
            for key, freq in frequency_data.items():
                label_in_key = False
                if analysis_col == self.horizon_analysis_col and self.use_max_range and label in self.range_labels:
                    max_label = self.get_max_range_label(frequency_data.keys())
                    label_in_key = label == max_label
                elif analysis_col == self.horizon_analysis_col:
                    label_in_key = label in key.split()
                elif analysis_col == self.reason_analysis_col:
                    label_in_key = label == key

                if label_in_key:
                    # if row[self.gen_code_pass_col] == '3/3':
                    if eval(row[self.gen_code_pass_col]) == 1:
                        weighted_pass += freq
                    weighted_total += freq
        return weighted_pass, weighted_total

    def update_reason_horizon_labels(self, label_mapping):
        for old_label, new_label in label_mapping.items():
            # Update self.labels_reasons
            if old_label in self.labels_reasons:
                self.labels_reasons.remove(old_label)
                if new_label not in self.labels_reasons:
                    self.labels_reasons.append(new_label)

            # Update self.labels_horizons
            if old_label in self.labels_horizons:
                self.labels_horizons.remove(old_label)
                if new_label not in self.labels_horizons:
                    self.labels_horizons.append(new_label)

    def overwrite_cols(self, df, label_mapping):
        if label_mapping is not None:
            # self.update_reason_horizon_labels(label_mapping)
            for index, row in df.iterrows():
                if isinstance(row[self.reason_analysis_col], str):
                    reason_dict = ast.literal_eval(row[self.reason_analysis_col])
                    df.at[index, self.reason_analysis_col] = {label_mapping.get(k, k): v for k, v in reason_dict.items()}
                if isinstance(row[self.horizon_analysis_col], str):
                    # import pdb; pdb.set_trace()
                    horizon_dict = ast.literal_eval(row[self.horizon_analysis_col])
                    # df.at[index, self.horizon_analysis_col] = {label_mapping.get(k, k): v for k, v in horizon_dict.items()}
                    df.at[index, self.horizon_analysis_col] = overwrite_first_word_in_dict_keys(horizon_dict, label_mapping)
                    # import pdb; pdb.set_trace()


        return df

    def analyze_csv(self, file_path, label_mapping=None, report_err_bar=True):
        df = pd.read_csv(file_path)
        model_name, gen_mode, gen_code_task, gen_code_time = self.extract_info_from_filename(file_path)
        print(model_name, gen_mode, gen_code_task, gen_code_time)  # Debugging line
        # If label_mapping is provided, overwrite the values in df[self.reason_analysis_col] and df[self.horizon_analysis_col]
        # import pdb; pdb.set_trace()
        if label_mapping:
            df = self.overwrite_cols(df, label_mapping)

        # Generalizing the pass ratio count based on the column and condition specified
        total_pass_cases = df.apply(lambda row: eval(row[self.gen_code_pass_col]) == 1, axis=1).sum()
        # import pdb; pdb.set_trace()
        summary = {
            'Model Name': model_name,
            'Generation Mode': gen_mode,
            'Code Task': gen_code_task,
            'All Pass Ratio': self.format_as_percentage(total_pass_cases, df.shape[0], report_err_bar=report_err_bar)
        }
       
        for label in self.labels_reasons:
            passed, total = self.get_pass_ratio(df, label, self.reason_analysis_col)
            summary[f'Pass Ratio {label.replace("_", " ")}'] = self.format_as_percentage(passed, total, report_err_bar=report_err_bar)
        for label in self.labels_horizons:
            passed, total = self.get_pass_ratio(df, label, self.horizon_analysis_col)
            summary[f'Pass Ratio {label.replace("_", " ")}'] = self.format_as_percentage(passed, total, report_err_bar=report_err_bar)
        return summary

    @staticmethod
    def extract_info_from_filename(filename):
        # Extract only the basename of the file to avoid path-related issues
        basename = os.path.basename(filename)
        
        # Adjusted Regex to match the filename format
        pattern = r"([^_]+)_(no_afterlines|with_afterlines|no_instruction)_(.+?)_(\d{2}_\d{2}_\d{2}_\d{2})"
        match = re.search(pattern, basename)
        if match:
            model_name = match.group(1)
            generation_mode = match.group(2)
            code_task = match.group(3).replace('_', ' ')
            generation_time = match.group(4)
            return model_name, generation_mode, code_task, generation_time
        else:
            return "Unknown", "Unknown", "Unknown", "Unknown"

    def analyze_results_in_folder(self, directory_path, label_mapping=None):
        summary_df = pd.DataFrame(columns=['Model Name', 'Generation Mode', 'Code Task', 'All Pass Ratio'] +
                                        [f'Pass Ratio {label.replace("_", " ")}' for label in self.labels_reasons + self.labels_horizons])
        for filename in os.listdir(directory_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory_path, filename)
                summary = self.analyze_csv(file_path, label_mapping=label_mapping)
                summary_df = summary_df.append(summary, ignore_index=True)
        return summary_df

# label_mapping = {
#     'If Condition': 'If Condition', #If Condition
#     'Elif Condition': 'If Condition',
#     'If Body': 'If Body',
#     'Elif Body': 'If Body',
#     'Else Reasoning': 'If Body',
#     'Loop Body': 'Loop',
#     'Define Stop Criteria': 'Loop',
# }


# label_mapping = {
#     # 'If Condition': 'test', #If Condition
#     # 'Elif Condition': 'test',
#     # 'If Body': 'test',
#     # 'Elif Body': 'test',
#     # 'Else Reasoning': 'test',
#     # 'Loop Body': 'test',
#     # 'Define Stop Criteria': 'test',
#     'Library': 'Function'
# }
# # label_mapping = None
# pd.set_option('display.max_columns', None)

# directory_path = './Analysis_Results/storage_server/Python_all_res/Completion/Test_Group_Labels/'
# analysis_instance = CodeAnalysis(weighted=False)  # Create an instance
# summary = analysis_instance.analyze_results_in_folder(directory_path, label_mapping=label_mapping)  # Call the method on the instance

# import pdb; pdb.set_trace()