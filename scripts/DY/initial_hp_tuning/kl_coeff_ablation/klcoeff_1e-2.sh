export N_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
export DATA_DIR="data/countdown-instruct"
export ROLLOUT_TP_SIZE=$N_GPUS
export VLLM_ATTENTION_BACKEND=XFORMERS

export USER_NAME=DY
export GROUP_NAME=qwen3b-instruct-hp-tuning
export EXPERIMENT_NAME=klcoeff_1e-2

# Create the log directory with the same pattern as tensorboard logs
export TODAY_DATE=$(date +"%m%d")
export LOG_DIR="tb_logs/${USER_NAME}/${TODAY_DATE}/${GROUP_NAME}/${EXPERIMENT_NAME}"
mkdir -p $LOG_DIR

ray stop --force && ray start --head --include-dashboard=True

python3 -m verl.trainer.main_ppo \
trainer.user_name=$USER_NAME \
trainer.group_name=$GROUP_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.critic_warmup=0 \
trainer.logger=['tensorboard'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=-1 \
trainer.test_freq=40 \
trainer.total_training_steps=200 \
algorithm.adv_estimator=grpo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=128 \
data.val_batch_size=1312 \
data.max_prompt_length=256 \
data.max_response_length=1024 \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_liger=False \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.01 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.n=8 \
actor_rollout_ref.rollout.temperature=1.0 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
algorithm.kl_ctrl.kl_coef=0.01 \
2>&1 | tee ${LOG_DIR}/verl_demo.txt