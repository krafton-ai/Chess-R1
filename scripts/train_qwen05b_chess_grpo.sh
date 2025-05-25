#!/bin/bash

# Environment variables
export N_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ROLLOUT_TP_SIZE=1  # Set tensor parallel
export VLLM_ATTENTION_BACKEND=XFORMERS  # Use XFORMERS for attention
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Define model and dataset
export DATA_DIR="data/lichess_10k_qwen_instruct_san_fen_legal_rule"
export BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# Experiment metadata
export USER_NAME="test"
export GROUP_NAME="test"
export EXPERIMENT_NAME="test"

timestamp=$(date +"%m%d-%H:%M")
DATA_NAME=$(basename "$DATA_DIR")       
BASE_MODEL_NAME=$(basename "$BASE_MODEL")
export EXPERIMENT_NAME="${timestamp}_${EXPERIMENT_NAME}_${DATA_NAME}_${BASE_MODEL_NAME}"
export LOG_DIR="outputs/${USER_NAME}/${GROUP_NAME}/${EXPERIMENT_NAME}"
export CHECKPOINT_DIR="outputs/${USER_NAME}/${GROUP_NAME}/${EXPERIMENT_NAME}/checkpoint"

trainer_args=" \
    trainer.user_name=$USER_NAME \
    trainer.group_name=$GROUP_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['tensorboard'] \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=40 \
    trainer.total_training_steps=200 \
    trainer.resume_from_path=False \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.default_hdfs_dir=$CHECKPOINT_DIR \
"

# batch_size: data.train_batch_size * actor.num_response
# data.train_batch_size: number of prompts
# actor.mini_batch_size: batch_size for gradient update
# actor.micro_batch_size_per_gpu: batch size per gpu for gradient accum
# updates per rollout: actor.epochs * (batch_size / actor.mini_batch_size)
data_args=" \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=1536 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.validate.num_rollouts_for_eval=1 \
"

reference_args=" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
"

algorithm_args=" \
    algorithm.gamma=1.0 \
"

TRAIN_ARGS="$trainer_args $data_args $actor_args $rollout_args $reference_args $algorithm_args"

##########################
# Train Model
ray stop --force && ray start --head --include-dashboard=True
python -m verl.trainer.main_grpo $TRAIN_ARGS 2>&1 | tee ${LOG_DIR}/verl_demo.log