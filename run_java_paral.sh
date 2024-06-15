#!/bin/bash

# Base directories for input and output
BASE_INPUT_DIR="./example_code/Java/COMP215"
OUTPUT_DIR="./Analysis_Results/runs_res"

# Get today's date in YYYY-MM-DD format
TODAY=$(date +%Y-%m-%d)
# Set the maximum number of concurrent jobs
MAX_JOBS=1
echo "Maximum jobs allowed concurrently: $MAX_JOBS"

# Set the wait time before the next job
WAIT_BEFORE_NEXT_JOB=0 #1200
echo "Wait time before the next job: $WAIT_BEFORE_NEXT_JOB seconds"

# Models and modes to loop through
models=("gpt-4-turbo" "gpt-3.5-turbo-0125")
modes=("with_afterlines")

# GPUs to use
GPU_IDS="" # GPU_IDS="2,3" - Use GPUs 2 and 3
IFS=',' read -ra gpus_to_use <<< "$GPU_IDS"
# Test cases
declare -a tests=(
    "A0/Test/FactorizationTester"
    "A1/Test/CounterTester"
    "A2/Test/DoubleVectorTester"
    "A3/Test/SparseArrayTester"
    "A4/Test/DoubleMatrixTester"
    "A5/Test/RNGTester"
    "A6/Test/TopKTester"
    "A7/Test/MTreeTester"
)

# Function to run analysis
run_analysis() {
    local java_file="$1"
    local json_file="$2"
    local gen_model="$3"
    local code_gen_mode="$4"
    local date_suffix="$5"
    local gpu_id="$6"
    local base_name=$(basename "$java_file" .java)
    # Output paths
    local stdout_path="${OUTPUT_DIR}/stdout_${base_name}_update_def_line_${gen_model}_${code_gen_mode}_${date_suffix}"
    local stderr_path="${OUTPUT_DIR}/stderr_${base_name}_update_def_line_${gen_model}_${code_gen_mode}_${date_suffix}"
    
    # Execute the Python script
    local command="python analyzer_demo_input_LNs.py $java_file $json_file --read_dependency_results --update_def_line --gen_model $gen_model --code_gen_mode $code_gen_mode > $stdout_path 2> $stderr_path"
    if [ -n "$gpu_id" ]; then
        command="CUDA_VISIBLE_DEVICES=$gpu_id $command"
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
    java_file="${BASE_INPUT_DIR}/${test}.java"
    json_file="${BASE_INPUT_DIR}/${test}.json"
    for model in "${models[@]}"; do
        for mode in "${modes[@]}"; do
            # Run analysis with dynamic date and GPU
            run_analysis "$java_file" "$json_file" "$model" "$mode" "$TODAY" "${gpus_to_use[$gpu_index]}"
            # Update GPU index for next job
            if [ ${#gpus_to_use[@]} -ne 0 ]; then
                let gpu_index=(gpu_index+1)%${#gpus_to_use[@]}
            fi
        done
    done
done
# Wait for all background jobs to complete
wait
echo "All analyses are complete."