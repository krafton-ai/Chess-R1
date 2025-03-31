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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import numpy as np
import torch
from tqdm import tqdm
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from collections import defaultdict
from copy import deepcopy
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.trainer.grpo.metrics import compute_data_metrics, compute_timing_metrics
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayResourcePool,
    RayWorkerGroup,
    RayClassWithInitArgs,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.grpo import core_algos
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import Tracking

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


def apply_kl_penalty(
    data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"
):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        # old_log_prob is actually the new log prob
        # ref log prob is the reference aka old log prob
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"],
            data.batch["ref_log_prob"],
            kl_penalty=kl_penalty,
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    if adv_estimator == "grpo":
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, 
            eos_mask=response_mask, 
            index=index,
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayGRPOTrainer(object):
    """
    This trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        train_reward_fn=None,
        val_reward_fn=None,
    ):
    
        self.tokenizer = tokenizer
        self.config = config
        self.train_reward_fn = train_reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping
            ), f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            self.kl_ctrl = core_algos.FixedKLController(
                kl_coef=config.algorithm.kl_coef
            )
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        self._validate_config()
        self._create_dataloader()
        self.global_steps = 0

    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = (
            config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        )
        assert (
            real_train_batch_size % n_gpus == 0
        ), f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # 2. Check whether mini_batch_size is divisible by micro_batch_size
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get(
                "ulysses_sequence_parallel_size", 1
            )
            micro_batch_size = (
                config.actor_rollout_ref.actor.micro_batch_size_per_gpu * n_gpus
            )
            assert (
                config.actor_rollout_ref.actor.mini_batch_size % micro_batch_size == 0
            )
            assert micro_batch_size * sp_size >= n_gpus

        # 3. Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            if (
                config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
                > 1
                or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1)
                > 1
            ):
                assert (
                    config.actor_rollout_ref.model.use_remove_padding
                ), "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("Validation datasets are sent to inference engines as a whole batch.")

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
        )
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data.train_num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get("image_key", "images"),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch
            batch_size=len(self.val_dataset),
            num_workers=self.config.data.val_num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        print(f"Size of train dataloader: {len(self.train_dataloader)}")

        # inject total_training_steps to optim_config. This is hacky.
        self.total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                self.total_training_steps
            )

    def init_workers(self):
        """
        Init resource pool and worker group
        """
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_scores = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]

            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "validate": True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                test_gen_batch_padded
            )
            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print(f"validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.validate.num_rollouts_for_eval,
                interleave=True,
            )
            
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )
        
        reward_tensor = (
            torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        )  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)
        
        for key, value in reward_metrics.items():
            metric_dict[f"val/{key}"] = value
        
        return metric_dict

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save,
        )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        # extract path to load checkpoint
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        # find global step folder
        if global_step_folder is None:
            print("Training from scratch")
            return 0
        else:
            # set global step
            self.global_steps = int(global_step_folder.split("global_step_")[-1])
            print(f"Setting global step to {self.global_steps}")
            print(f"Resuming from {global_step_folder}")

            # load actor
            actor_path = os.path.join(global_step_folder, "actor")
            self.actor_rollout_wg.load_checkpoint(
                actor_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

            # load dataloader,
            dataloader_local_path = os.path.join(global_step_folder, "data.pt")
            if os.path.exists(dataloader_local_path):
                dataloader_state_dict = torch.load(dataloader_local_path)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
            else:
                print(
                    f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch data"
                )

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = (
            batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        )  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition]
        )
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of GRPO with dynamic sampling from DAPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the GRPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        
        logger = Tracking(
            user_name=self.config.trainer.user_name,
            group_name=self.config.trainer.group_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        # TODO: GET RID OF THIS COMMENT
        # val_metrics = self._validate()
        # pprint(f"Initial validation metrics: {val_metrics}")
        # logger.log(data=val_metrics, step=self.global_steps)
        # self.global_steps += 1
        
        if self.total_training_steps == 0:
            return

        collected_batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        # Create iterator for dataloader to manage iteration manually
        dataloader_iter = iter(self.train_dataloader)

        # Start training with tqdm progress bar
        with tqdm(total=self.total_training_steps, desc="Training", unit="step") as pbar:
            # While loop based on steps instead of nested epoch/batch loops
            while self.global_steps < self.total_training_steps:
                # Get next new_gen_batch with handling for dataloader exhaustion
                try:
                    batch_dict = next(dataloader_iter)
                except StopIteration:
                    # Reinitialize the dataloader iterator when it's exhausted
                    dataloader_iter = iter(self.train_dataloader)
                    batch_dict = next(dataloader_iter)
                print(f"step {self.global_steps}")

                metrics = {}
                timing_raw = {}

                new_gen_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                gen_batch = new_gen_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            gen_batch
                        )

                    new_gen_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_gen_batch.batch))],
                        dtype=object,
                    )
                    # repeat to align with repeated responses in rollout
                    new_gen_batch = new_gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    new_gen_batch = new_gen_batch.union(gen_batch_output)

                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_gen_batch)
                            new_gen_batch = new_gen_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, correct_seq_tensor, reward_metrics = self.train_reward_fn(new_gen_batch)
                        metrics.update(reward_metrics)
                        new_gen_batch.batch["token_level_scores"] = deepcopy(reward_tensor)
                        new_gen_batch.batch["correct_sequences"] = deepcopy(correct_seq_tensor)

                        # compute rewards. apply_kl_penalty to the rewards if available
                        if not self.config.actor_rollout_ref.actor.get(
                            "use_kl_loss", False
                        ):
                            new_gen_batch, kl_metrics = apply_kl_penalty(
                                new_gen_batch,
                                kl_ctrl=self.kl_ctrl,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_gen_batch.batch["token_level_rewards"] = deepcopy(new_gen_batch.batch[
                                "token_level_scores"
                            ])
                    
                    with _timer("dynamic_sampling", timing_raw):
                        if not self.config.algorithm.dynamic_sampling.enable and not self.config.algorithm.overlong_buffer.enable:
                            collected_batch = new_gen_batch
                        else:
                            # NOTE: When prompts after filtering is less than train batch size, we skip to the next generation batch
                            if self.config.algorithm.dynamic_sampling.enable:  
                                num_samples = new_gen_batch.batch['token_level_scores'].shape[0]

                                batch_per_seq_final_rewards = new_gen_batch.batch['token_level_scores'].sum(dim=-1).numpy()

                                # Collect the sequence reward for each trajectory
                                prompt_uid2rew_vals = defaultdict(list)
                                for uid, rew_val in zip(new_gen_batch.non_tensor_batch['uid'], batch_per_seq_final_rewards):
                                    prompt_uid2rew_vals[uid].append(rew_val)

                                prompt_uid2rew_std = {}
                                for prompt_uid, rew_vals in prompt_uid2rew_vals.items():
                                    prompt_uid2rew_std[prompt_uid] = np.std(rew_vals)

                                # Different from DAPO, we're gonna exclude all samples which have same values over all samples
                                kept_rew_prompt_uids = [uid for uid, std in prompt_uid2rew_std.items() if std > 0]
                                metrics["train/num_dynsamp_discard_samples"] = num_samples - len(kept_rew_prompt_uids) * self.config.actor_rollout_ref.rollout.n
                            else:
                                kept_rew_prompt_uids = list(set(new_gen_batch.non_tensor_batch['uid']))
                            
                            # Modified Overlong Reward Shaping from DAPO
                            # Unlike from the original implentation in DAPO, we drop samples which have been cutoff due to max generation length.
                            # i.e., we don't use the soft penalty. we only use the hard penalty.
                            if self.config.algorithm.overlong_buffer.enable:
                                num_samples = new_gen_batch.batch['token_level_scores'].shape[0]
                                # Maximum allowed length for responses from the overlong buffer
                                max_response_length = self.config.algorithm.overlong_buffer.max_response_length
                                
                                # Extract the prompt IDs and determine the prompt length
                                prompt_ids = new_gen_batch.batch["prompts"]
                                prompt_length = prompt_ids.shape[-1]
                                
                                # Compute valid response lengths using the attention mask, excluding the prompt part
                                valid_response_lengths = new_gen_batch.batch["attention_mask"][:, prompt_length:].sum(axis=-1)
                                
                                # Create a mapping from each unique prompt ID to its list of valid response lengths
                                prompt_uid_to_lengths = defaultdict(list)
                                for uid, length in zip(new_gen_batch.non_tensor_batch["uid"], valid_response_lengths):
                                    prompt_uid_to_lengths[uid].append(length.item())
                                
                                # Filter out prompt IDs that contain any responses that hit the max response length (i.e., are overlong)
                                kept_length_prompt_uids = [
                                    uid for uid, lengths in prompt_uid_to_lengths.items()
                                    if all(l < max_response_length for l in lengths)
                                ]
                                
                                # Compute the intersection of kept_rew_prompt_uids and kept_length_prompt_uids
                                final_kept_prompt_uids = list(set(kept_rew_prompt_uids) & set(kept_length_prompt_uids))
                                metrics["train/num_overlong_discard_samples"] = num_samples - len(kept_length_prompt_uids) * self.config.actor_rollout_ref.rollout.n
                            else:
                                final_kept_prompt_uids = kept_rew_prompt_uids

                            kept_traj_idxs = []
                            for idx, traj_from_prompt_uid in enumerate(new_gen_batch.non_tensor_batch['uid']):
                                if traj_from_prompt_uid in final_kept_prompt_uids:
                                    kept_traj_idxs.append(idx)
                            
                            num_prompt_in_batch += len(final_kept_prompt_uids)
                    
                            new_gen_batch = new_gen_batch[kept_traj_idxs]

                            if collected_batch is None:
                                collected_batch = new_gen_batch
                            else:
                                collected_batch = DataProto.concat([collected_batch, new_gen_batch])

                            prompt_bsz = self.config.data.train_batch_size
                            if num_prompt_in_batch < prompt_bsz:
                                print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                                max_num_gen_batches = self.config.algorithm.dynamic_sampling.max_num_gen_batches
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f'{num_gen_batches=}. Keep generating...')
                                    continue
                                else:
                                    raise ValueError(
                                        f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                    )
                            else:
                                # Align the batch
                                collected_batch = DataProto(
                                    batch=collected_batch.batch, 
                                    non_tensor_batch=collected_batch.non_tensor_batch, 
                                    meta_info=collected_batch.meta_info
                                )
                                traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                                collected_batch = collected_batch[:traj_bsz]
                                
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    collected_batch = DataProto(
                                    batch=collected_batch.batch, 
                                    non_tensor_batch=collected_batch.non_tensor_batch, 
                                    meta_info=collected_batch.meta_info
                                )
                    self._balance_batch(collected_batch, metrics=metrics)

                    # compute global_valid tokens
                    collected_batch.meta_info["global_token_num"] = torch.sum(
                        collected_batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(collected_batch)
                        collected_batch = collected_batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                collected_batch
                            )
                            collected_batch = collected_batch.union(ref_log_prob)

                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        collected_batch = compute_advantage(
                            collected_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )

                    # update actor
                    with _timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(collected_batch)
                    actor_output_metrics = reduce_metrics(
                        actor_output.meta_info["metrics"]
                    )
                    metrics.update(actor_output_metrics)

                    # validate
                    #TODO: UNCOMMENT THIS!!
                    # if (
                    #     self.val_reward_fn is not None
                    #     and self.config.trainer.test_freq > 0
                    #     and self.global_steps % self.config.trainer.test_freq == 0
                    # ):
                    #     with _timer("testing", timing_raw):
                    #         val_metrics: dict = self._validate()
                    #     metrics.update(val_metrics)

                    if (
                        self.config.trainer.save_freq > 0
                        and self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=collected_batch))
                metrics.update(
                    compute_timing_metrics(batch=collected_batch, timing_raw=timing_raw)
                )

                metrics["train/num_gen_batches"] = num_gen_batches
                collected_batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                pbar.update(1)

        #TODO: UNCOMMENT THIS!!
        # # perform validation after training
        # if self.val_reward_fn is not None:
        #     val_metrics = self._validate()
        #     pprint(f"Final validation metrics: {val_metrics}")
        #     logger.log(data=val_metrics, step=self.global_steps)
        if (
            self.config.trainer.save_freq > 0
            and (self.global_steps - 1) % self.config.trainer.save_freq != 0
        ):
            with _timer("save_checkpoint", timing_raw):
                self._save_checkpoint()
        return
