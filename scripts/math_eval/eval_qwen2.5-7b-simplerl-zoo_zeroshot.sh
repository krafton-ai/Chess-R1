#!/bin/bash

# hugging-face model
timestamp=$(date +"%m%d-%H:%M")
export OUTPUT_DIR="outputs/math_eval/${timestamp}_Qwen2.5-7B-SimpleRL-Zoo/"
export MODEL_NAME="hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo"
export TOKENIZERS_PARALLELISM=false

TASKS=(
    # "mmlu" "gpqa_diamond" 
    # "aime24" "gsm8k" "math500" "minervamath" "olympiadbench_math_en"
    # "minervamath"
    "gsm8k" "math500"
    # "livecodebench_easy" "livecodebench_medium" "livecodebench_hard"
)

backend_args="tensor_parallel_size=1,gpu_memory_utilization=0.8"
sampling_params="max_tokens=16000,temperature=1.0,top_p=0.95"

# GPU assignment tracking
export N_GPUS=4
GPU_IDS=(0 1 2 3)
# export N_GPUS=3
# GPU_IDS=(1 2 3)
PIDS=()
GPU_ASSIGN=()
for ((i=0; i<N_GPUS; i++)); do
    GPU_ASSIGN[i]=0
done

# Function to run a task
run_task() {
    local model_path="$1"
    local task="$2"
    local gpu_index="$3"
    local gpu_id="${GPU_IDS[$gpu_index]}"

    echo "Running on GPU $gpu_id: Model=$model_path, Task=$task"

    # Set CUDA_VISIBLE_DEVICES to assign the process to the selected GPU and run in background
    # If you want to save generated output, please pass --save-result / --no-save-result flag
    # In skythought, n means Best-of-N Evaluation
    # https://github.com/NovaSky-AI/SkyThought/tree/main/skythought/evals
    CUDA_VISIBLE_DEVICES=$gpu_id skythought evaluate \
        --model "$model_path" \
        --task "$task" \
        --backend "vllm" \
        --backend-args "$backend_args" \
        --sampling-params "$sampling_params" \
        --n 1 \
        --result-dir "$OUTPUT_DIR" \
        --save-result &

    # Store process ID and mark GPU as busy
    PIDS[$gpu_index]=$!
    GPU_ASSIGN[$gpu_index]=1

    sleep 1
}

# Launch tasks dynamically as GPUs become available
task_idx=0
total_tasks=${#TASKS[@]}

while (( task_idx < total_tasks )); do
    # Find an available GPU
    available_gpu=-1
    for ((gpu_index=0; gpu_index<N_GPUS; gpu_index++)); do
        if (( GPU_ASSIGN[gpu_index] == 0 )); then
            available_gpu=$gpu_index
            break
        fi
    done

    if (( available_gpu >= 0 )); then
        task="${TASKS[$task_idx]}"
        run_task "$MODEL_NAME" "$task" "$available_gpu"
        (( task_idx++ ))
    else
        # No free GPU available, poll for finished tasks
        finished_found=false
        for ((i=0; i<N_GPUS; i++)); do
            if (( GPU_ASSIGN[i] == 1 )); then
                # kill -0: alive check
                if ! kill -0 "${PIDS[i]}" 2>/dev/null; then
                    echo "Task on GPU ${GPU_IDS[i]} finished."
                    GPU_ASSIGN[i]=0
                    finished_found=true
                fi
            fi
        done
        # If no task has finished, wait a bit before checking again
        if ! $finished_found; then
            sleep 1
        fi
    fi
done

# Wait for any remaining processes to complete
wait

echo "All evaluations completed!"
