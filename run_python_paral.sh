#!/bin/bash
# Base directories for input and output
BASE_INPUT_DIR="./example_code/Python"
OUTPUT_DIR="./Analysis_Results/runs_res"
# Get today's date in YYYY-MM-DD format
TODAY=$(date +%Y-%m-%d)
# Set the maximum number of concurrent jobs
MAX_JOBS=1
echo "Maximum jobs allowed concurrently: $MAX_JOBS"

# Set the wait time before the next job
WAIT_BEFORE_NEXT_JOB=0 #600
echo "Wait time before the next job: $WAIT_BEFORE_NEXT_JOB seconds"

# Closed source model list
closed_source_model_list=("claude-3-haiku-20240307" "claude-3-sonnet-20240229" "claude-3-opus-20240229" "gpt-4-turbo" "gpt-3.5-turbo-0125")

# Open source model list
open_source_models_list=("deepseek-coder-1.3b-instruct" "deepseek-coder-7b-instruct" "phi-3-mini-4k" "Meta-Llama-3-8B-Instruct" "Meta-Llama-3-70B-Instruct")

# Models and modes to loop through
models=("gpt-3.5-turbo-0125")
modes=("no_afterlines")

# GPUs to use
GPU_IDS="" # GPU_IDS="2,3" - Use GPUs 2 and 3
IFS=',' read -ra gpus_to_use <<< "$GPU_IDS"
# Test cases
declare -a tests=(
    "Image_Filtering/Image_Filtering"
    "Credit_Scoring_Fairness/Credit_Scoring_Fairness"
    "Image_Transformation/Image_Transformation"
    "simplex_method/simplex_method"
    "RL_Motion_Planning/RL_Motion_Planning"
    "GAN_model/GAN_model"
    "Timeseries_Clustering/Timeseries_Clustering"
    # Add more test cases as needed
)
# Function to run analysis
run_analysis() {
    local python_file="$1"
    local json_file="$2"
    local gen_model="$3"
    local code_gen_mode="$4"
    local date_suffix="$5"
    local gpu_id="$6"
    local base_name=$(basename "$python_file" .py)
    # Output paths
    local stdout_path="${OUTPUT_DIR}/stdout_${base_name}_update_def_line_${gen_model}_${code_gen_mode}_${date_suffix}"
    local stderr_path="${OUTPUT_DIR}/stderr_${base_name}_update_def_line_${gen_model}_${code_gen_mode}_${date_suffix}"
    # Check if the model is in the closed source list
    if [[ " ${closed_source_model_list[@]} " =~ " ${gen_model} " ]]; then
        # For closed source models, do not use GPU
        command="python close_source_model_gen.py $python_file $json_file --read_dependency_results --update_def_line --gen_model $gen_model --code_gen_mode $code_gen_mode > $stdout_path 2> $stderr_path"
    else
        # For open source models, use GPU if specified
        if [ -n "$gpu_id" ]; then
            command="CUDA_VISIBLE_DEVICES=$gpu_id python -m open_source_model_gen.open_source_code_gen $python_file $json_file --gen_model $gen_model --code_gen_mode $code_gen_mode > $stdout_path 2> $stderr_path"
        else
            command="python -m open_source_model_gen.open_source_code_gen $python_file $json_file --gen_model $gen_model --code_gen_mode $code_gen_mode > $stdout_path 2> $stderr_path"
        fi
    fi
    echo "Running command: $command"
    eval $command &
    # Manage job control to limit to MAX_JOBS concurrent jobs
    while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
    # Wait for a certain number of seconds before the next job
    sleep $WAIT_BEFORE_NEXT_JOB
}
# Iterate through each test case and combination of model and mode
gpu_index=0
for test in "${tests[@]}"; do
    python_file="${BASE_INPUT_DIR}/${test}.py"
    json_file="${BASE_INPUT_DIR}/${test}.json"
    for model in "${models[@]}"; do
        for mode in "${modes[@]}"; do
            # Run analysis with dynamic date and GPU
            run_analysis "$python_file" "$json_file" "$model" "$mode" "$TODAY" "${gpus_to_use[$gpu_index]}"
            # Update GPU index for next job
            if [ ${#gpus_to_use[@]} -ne 0 ] && [[ ! " ${closed_source_model_list[@]} " =~ " ${model} " ]]; then
                let gpu_index=(gpu_index+1)%${#gpus_to_use[@]}
            fi
        done
    done
done
# Wait for all background jobs to complete
wait
echo "All analyses are complete."