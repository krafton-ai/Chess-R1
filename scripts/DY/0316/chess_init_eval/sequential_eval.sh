#!/bin/bash

# Common environment variables
export N_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ROLLOUT_TP_SIZE=1        # Tensor parallel setting
export VLLM_ATTENTION_BACKEND=XFORMERS  # Use XFORMERS for attention

export USER_NAME="DY"
export GROUP_NAME="chess_init_eval"

# Arrays for configuration differences (multi-line for readability)
BASE_MODELS=(
    # "Qwen/Qwen2.5-7B"
    # "Qwen/Qwen2.5-7B"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-8B"
    # "meta-llama/Llama-3.1-8B"

)

DATA_DIRS=(
    # "data/dev_chess_base_with_legal_move"
    # "data/dev_chess_base_wo_legal_move"
    # "data/dev_chess_qwen_instruct_with_legal_move"
    # "data/dev_chess_qwen_instruct_wo_legal_move"
    "data/dev_chess_llama_instruct_with_legal_move"
    "data/dev_chess_llama_instruct_with_legal_move"
    # "data/dev_chess_llama_instruct_wo_legal_move"
    "data/dev_chess_base_with_legal_move"
    # "data/dev_chess_base_wo_legal_move"
)

EXPERIMENT_NAMES=(
    # "eval-qwen-base-w-legal"
    # "eval-qwen-base-wo-legal"
    # "eval-qwen-instruct-w-legal"
    # "eval-qwen-instruct-wo-legal"
    "eval-llama3-1-8B-instruct-w-legal"
    "eval-llama3-0-8B-instruct-w-legal"
    # "eval-llama-instruct-wo-legal"
    "eval-llama3-1-8N-base-w-legal"
    # "eval-llama-base-wo-legal"
)

TEMPERATURES=(
    0.0
    0.6
    1.0
)

# Outer loop: iterate over experiment c=onfigurations
for i in "${!BASE_MODELS[@]}"; do
    export BASE_MODEL="${BASE_MODELS[$i]}"
    export DATA_DIR="${DATA_DIRS[$i]}"
    base_experiment_name="${EXPERIMENT_NAMES[$i]}"

    # Inner loop: iterate over temperature values
    for temp in "${TEMPERATURES[@]}"; do
        # Append temperature to the experiment name
        export EXPERIMENT_NAME="${base_experiment_name}_temp${temp}"

        export TODAY_DATE=$(date +"%m%d")
        export LOG_DIR="tb_logs/${USER_NAME}/${TODAY_DATE}/${GROUP_NAME}/${EXPERIMENT_NAME}"
        mkdir -p "$LOG_DIR"
        LOG_FILE="${LOG_DIR}/verl_demo.txt"

        # Build common argument strings
        trainer_args=" \
            trainer.user_name=$USER_NAME \
            trainer.group_name=$GROUP_NAME \
            trainer.experiment_name=$EXPERIMENT_NAME \
            trainer.logger=['tensorboard'] \
            trainer.n_gpus_per_node=$N_GPUS \
            trainer.nnodes=1 \
            trainer.save_freq=500 \
            trainer.test_freq=40 \
            trainer.total_training_steps=1000 \
        "

        data_args=" \
            data.train_files=$DATA_DIR/train.parquet \
            data.val_files=$DATA_DIR/test.parquet \
            data.train_batch_size=128 \
            data.max_prompt_length=768 \
            data.max_response_length=2048 \
        "

        actor_args=" \
            actor_rollout_ref.model.path=$BASE_MODEL \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.epochs=1 \
            actor_rollout_ref.actor.mini_batch_size=32 \
            actor_rollout_ref.actor.micro_batch_size_per_gpu=2 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.001 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        "

        rollout_args=" \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
            actor_rollout_ref.rollout.n=1 \
            actor_rollout_ref.rollout.temperature=$temp \
        "

        reference_args=" \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
        "

        algorithm_args=" \
            algorithm.gamma=1.0 \
        "

        ARGS="$trainer_args $data_args $actor_args $rollout_args $reference_args $algorithm_args"

        # Restart Ray for each experiment
        ray stop --force && ray start --head --include-dashboard=True

        echo "-----------------------------"
        echo "Running experiment: $EXPERIMENT_NAME"
        echo "Model: $BASE_MODEL"
        echo "Data directory: $DATA_DIR"
        echo "Temperature: $temp"
        echo "Logging to: $LOG_FILE"
        echo "-----------------------------"

        # Run the evaluation command
        python -m verl.trainer.eval_grpo $ARGS 2>&1 | tee "$LOG_FILE"
    done
done
