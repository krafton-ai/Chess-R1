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
import jax
import chess
import torch
import time
import numpy as np
from collections import defaultdict
from verl import DataProto
from verl.utils.reward_score import gsm8k, math, multiply, countdown, understand_r1_math_grader
from jax import random as jrandom

import spacy
import spacy_fastlang
import fasttext
# Override fastText's internal warning function to a no-op.
fasttext.FastText.eprint = lambda *args, **kwargs: None

from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils as sr_utils
from searchless_chess.src.engines import engine
from searchless_chess.src.engines import neural_engines

def _select_rm_score_fn(data_source):
    if data_source == "openai/gsm8k":
        return gsm8k.compute_score
    elif data_source == "lighteval/MATH":
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    elif "math_12k" in data_source:
        return understand_r1_math_grader.compute_score
    else:
        raise NotImplementedError


class RewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, rew_configs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.rew_configs = rew_configs
    
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        st_time = time.time()

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        correct_seq_tensor = torch.zeros(*(data.batch.batch_size), dtype=torch.int32)

        already_print_data_sources = {}
        agg_reward_logs = defaultdict(float)
        example_texts = []

        for i in range(len(data)):
            data_item = data[i] # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            # select rm_score
            data_source = data_item.non_tensor_batch["data_source"]
            data_template_type = data_item.non_tensor_batch["template_type"]
            compute_score_fn = _select_rm_score_fn(data_source)

            score, correct_seq, reward_logs = compute_score_fn(
                    solution_str=sequences_str, 
                    ground_truth=ground_truth, 
                    method=data_template_type,
                    format_reward=self.rew_configs.format_reward,
                    answer_reward=self.rew_configs.answer_reward,
                )
            
            reward_tensor[i, valid_response_length - 1] = score
            correct_seq_tensor[i] = correct_seq

            # aggregate logging metrics
            for key, value in reward_logs.items():
                agg_reward_logs[key] += value

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                example_texts.append(sequences_str)
        
        # normalize aggregate logging metrics
        normalized_agg_reward_logs = {}
        for key in agg_reward_logs.keys():
            normalized_agg_reward_logs[f'reward/{key}'] = agg_reward_logs[key] / len(data)
        
        # add a single sequence_str as an example
        normalized_agg_reward_logs[f'generation/text'] = "\n\n".join(example_texts)

        return reward_tensor, correct_seq_tensor, normalized_agg_reward_logs
