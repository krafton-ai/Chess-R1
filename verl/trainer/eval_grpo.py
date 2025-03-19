# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import ray
import hydra
from pprint import pprint
from verl.trainer.grpo.ray_trainer import RayGRPOTrainer


@hydra.main(config_path="config", config_name="grpo_trainer", version_base=None)
def main(config):
    run_grpo(config)


def run_grpo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true", 
                    "NCCL_DEBUG": "WARN",
                    # Make sure VLLM_ATTENTION_BACKEND is set in the Ray runtime environment
                    "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND", "XFORMERS"),
                    # Set to "1" to allow debugging
                    "RAY_DEBUG": "1"
                }
            }
        )
    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):

    # instantiate configuration
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # download the model checkpoint
    from verl.utils.fs import copy_to_local

    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    # define actor class as ray worker
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        from verl.trainer.grpo.workers.fsdp_workers import ActorRolloutRefWorker
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.grpo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # instantiate reward function
    # - for model-based rm, we call a reward model
    # - for rule-based rm, we directly call a reward score
    # - for code related prompt, we send to a sandbox if there are test cases
    from verl.trainer.grpo.reward_manager import RewardManager

    train_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # define reward model class as ray worker
    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.trainer.grpo.fsdp_workers import RewardModelWorker

            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        else:
            raise NotImplementedError

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        },
        mapping=mapping,
    )

    trainer = RayGRPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        train_reward_fn=train_reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    val_metrics = trainer._validate()
    print(f"Initial validation metrics: {val_metrics}")
    pprint(f"Initial validation metrics: {val_metrics}")

if __name__ == "__main__":
    main()
