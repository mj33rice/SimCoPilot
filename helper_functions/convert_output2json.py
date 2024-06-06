import json
import os
import argparse

def convert_txt_to_json(txt_file, json_file, key):
    # Read the text file
    with open(txt_file, 'r') as f:
        content = f.read()

    # Initialize data
    data = {}

    # Read the existing JSON file if it exists
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

    # Add the new content to the specified key
    data[key] = content

    # Add "test_case1" with an empty string
    data["test_case1"] = ""

    # Write the updated data back to the JSON file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# # Usage:
# output_file = '../example_code/Python/RL_Motion_Planning/run1.txt'
# json_file = '../example_code/Python/RL_Motion_Planning/RL_Motion_Planning.json'
# convert_txt_to_json(output_file, json_file, 'expected_output1')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert txt to json.')
    parser.add_argument('output_file', type=str, help='The path to the output file.')
    parser.add_argument('json_file', type=str, help='The path to the json file.')
    args = parser.parse_args()

    convert_txt_to_json(args.output_file, args.json_file, 'expected_output1')