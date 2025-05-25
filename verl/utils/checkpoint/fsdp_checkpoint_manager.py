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

import ray
import os

import warnings
from typing import Union
import torch
import torch.distributed
from torch.distributed._tensor import DTensor

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig

from verl.utils.fs import copy_to_local, is_non_local

from transformers import PreTrainedTokenizer, ProcessorMixin

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer/processor and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        **kwargs,
    ):

        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.",
                DeprecationWarning,
            )
            processing_class = kwargs.pop("tokenizer")

        super().__init__(model, optimizer, lr_scheduler, processing_class)

    def load_checkpoint(self, path=None, del_local_after_load=False, *args, **kwargs):
        if path is None:
            return

        # every rank download its own checkpoint
        remote_model_path = os.path.join(
            path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt"
        )
        remote_optim_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt"
        )
        remote_extra_state_path = os.path.join(
            path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
        )
        print(
            f"[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}"
        )
        local_model_path = copy_to_local(remote_model_path)
        local_optim_path = copy_to_local(remote_optim_path)
        local_extra_state_path = copy_to_local(remote_extra_state_path)

        model_state_dict = torch.load(local_model_path)
        optimizer_state_dict = None
        extra_state_dict = {"lr_scheduler": None}
        if os.path.exists(local_optim_path):
            optimizer_state_dict = torch.load(local_optim_path)
        if os.path.exists(local_extra_state_path):
            extra_state_dict = torch.load(local_extra_state_path)

        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                (
                    os.remove(local_extra_state_path)
                    if is_non_local(local_extra_state_path)
                    else None
                )
            except Exception as e:
                print(
                    f"[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored"
                )

        lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(
            self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
        ):
            self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None and optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        if "rng" in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict["rng"])

        if self.lr_scheduler is not None and extra_state_dict.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def save_checkpoint(
        self,
        local_path: str,
        global_step: int,
        remove_previous_ckpt=False,
        *args,
        **kwargs,
    ):
        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        # TODO: shall we remove previous ckpt every save?
        if remove_previous_ckpt:
            self.remove_previous_save_local_path()
        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FSDP.state_dict_type(
                self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
            ):
                model_state_dict = self.model.state_dict()
                if self.optimizer is not None:
                    optimizer_state_dict = self.optimizer.state_dict()
                else:
                    optimizer_state_dict = None
                if self.lr_scheduler is not None:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict()
                else:
                    lr_scheduler_state_dict = None

                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "rng": self.get_rng_state(),
                }
                model_path = os.path.join(
                    local_path,
                    f"model_world_size_{self.world_size}_rank_{self.rank}.pt",
                )
                optim_path = os.path.join(
                    local_path,
                    f"optim_world_size_{self.world_size}_rank_{self.rank}.pt",
                )
                extra_path = os.path.join(
                    local_path,
                    f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt",
                )

                print(
                    f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}"
                )
                print(
                    f"[rank-{self.rank}]: Saving checkpoint to {os.path.abspath(model_path)}"
                )
                print(
                    f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}"
                )
                torch.save(model_state_dict, model_path)
                torch.save(
                    optimizer_state_dict, optim_path
                )  # TODO: address optimizer is None
                torch.save(extra_state_dict, extra_path)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.rank == 0:
            hf_local_path = os.path.join(local_path, "huggingface")
            os.makedirs(hf_local_path, exist_ok=True)
            self.model._fsdp_wrapped_module.config.save_pretrained(hf_local_path)
            self.processing_class.save_pretrained(hf_local_path)

        torch.distributed.barrier()

        self.previous_save_local_path = local_path

    def shrink_perturb(self, path1: str, path2: str, alpha: float, tmp_path: str = "/tmp/blended_ckpt") -> None:
        """
        Blends model parameters from two checkpoints:
        new_param = alpha * param1 + (1 - alpha) * param2

        Saves the blended parameters as a new checkpoint and loads it into self.model.
        """

        def load_full_state(path: str):
            model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
            local_model_path = copy_to_local(model_path)
            return torch.load(local_model_path, map_location="cpu")

        print(f"[rank-{self.rank}]: Loading checkpoint 1 from {path1}")
        state_dict_1 = load_full_state(path1)

        print(f"[rank-{self.rank}]: Loading checkpoint 2 from {path2}")
        state_dict_2 = load_full_state(path2)

        # Blend parameters
        print(f"[rank-{self.rank}]: Blending state_dicts with alpha={alpha}")
        blended_state_dict = {}
        for k in state_dict_1:
            blended_state_dict[k] = alpha * state_dict_1[k] + (1 - alpha) * state_dict_2[k]

        # Save blended model temporarily
        os.makedirs(tmp_path, exist_ok=True)
        blended_model_path = os.path.join(tmp_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        torch.save(blended_state_dict, blended_model_path)
        print(f"[rank-{self.rank}]: Saved blended checkpoint to {blended_model_path}")

        # Reload using load_checkpoint (reuses FSDP logic)
        self.load_checkpoint(tmp_path)
