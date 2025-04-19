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
from verl.utils.reward_score import gsm8k, math, multiply, countdown, think_chess, answer_chess, lichess, \
                                    chess_best_move, chess_comparison, chess_mechanics, deepmind_lichess
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
    elif "answer_chess" in data_source:
        return answer_chess.compute_score
    elif "chess_fen" in data_source:
        return think_chess.compute_score
    elif "deepmind_lichess" in data_source:
        return deepmind_lichess.compute_score
    elif "lichess" in data_source:
        return lichess.compute_score
    elif "chess_best_move" in data_source:
        return chess_best_move.compute_score
    elif "chess_comparison" in data_source:
        return chess_comparison.compute_score
    elif "chess_mechanics" in data_source:
        return chess_mechanics.compute_score
    else:
        raise NotImplementedError


class RewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, rew_configs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.rew_configs = rew_configs

        # language detector
        spacy.require_gpu()
        self.lg_detector = spacy.load("en_core_web_lg")
        for pipe_name in self.lg_detector.pipe_names: # Remove all existing pipes
            self.lg_detector.remove_pipe(pipe_name)
        self.lg_detector.add_pipe("language_detector")

        self.chess_model, self.return_buckets_values = self._setup_chess_model(self.rew_configs)
        
    def _setup_chess_model(self, rew_configs):
        chess_model_type = rew_configs.chess_model_type
        num_return_buckets = 128
        match chess_model_type:
            case 'action_value':
                output_size = num_return_buckets
            case 'behavioral_cloning':
                output_size = sr_utils.NUM_ACTIONS
            case 'state_value':
                output_size = num_return_buckets
            case _:
                raise ValueError(f'Unknown chess model type: {chess_model_type}')

        predictor_config = transformer.TransformerConfig(
            vocab_size=sr_utils.NUM_ACTIONS,
            output_size=output_size,
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
            num_heads=8,
            num_layers=16,
            embedding_dim=1024,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )

        predictor = transformer.build_transformer_predictor(config=predictor_config)

        # Load the predictor parameters
        checkpoint_dir = os.path.join(
            os.getcwd(),
            f'searchless_chess/checkpoints/270M',
        )
        dummy_params = predictor.initial_params(
            rng=jrandom.PRNGKey(0),
            targets=np.zeros((1, 1), dtype=np.uint32),
        )
        params = training_utils.load_parameters(
            checkpoint_dir=checkpoint_dir,
            params=dummy_params,
            use_ema_params=True,
            step=-1,
        )

        _, return_buckets_values = sr_utils.get_uniform_buckets_edges_values(
            num_return_buckets
        )

        predict_fn = neural_engines.wrap_predict_fn(predictor, params, batch_size=256)
        
        neural_engine = neural_engines.ENGINE_FROM_POLICY[chess_model_type](
            return_buckets_values=return_buckets_values,
            predict_fn=predict_fn,
            temperature=0.005,
        )

        return neural_engine, return_buckets_values
    
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        st_time = time.time()

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        if self.rew_configs.qvalue_reward_scaler > 0:
            # === Precompute q-values for all final states in batch ===
            # Extract the FEN based board state from each sample's ground truth.
            board_list = []
            for i in range(len(data)):
                ground_truth = data[i].non_tensor_batch["reward_model"]["ground_truth"]
                fen_state = ground_truth.get("board_fen", "")
                board_list.append(chess.Board(fen_state))
            precomputed_chess_results= self.chess_model.analyse_batch(boards=board_list)
            # # Reshape the results back to (B, qvalue_dim)
            precomputed_chess_qvalues_list = [np.inner(np.exp(result['log_probs']), self.return_buckets_values) for result in precomputed_chess_results]
        else:
            precomputed_chess_qvalues_list = [0.0 for i in range(len(data))]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        correct_seq_tensor = torch.zeros(*(data.batch.batch_size), dtype=torch.int32)

        already_print_data_sources = {}
        agg_reward_logs = defaultdict(float)
        example_texts = []

        for i in range(len(data)):
            data_item = data[i] # DataProtoItem
            precomputed_chess_qvalues = precomputed_chess_qvalues_list[i]

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

            ground_truth_dict = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            # select rm_score
            data_source = data_item.non_tensor_batch["data_source"]
            compute_score_fn = _select_rm_score_fn(data_source)

            score, correct_seq, reward_logs = compute_score_fn(
                    solution_str=sequences_str, 
                    ground_truth_dict=ground_truth_dict, 
                    lg_detector=self.lg_detector,
                    chess_model_qvalues=precomputed_chess_qvalues,
                    format_reward=self.rew_configs.format_reward,
                    english_reward=self.rew_configs.english_reward,
                    answer_reward=self.rew_configs.answer_reward,
                    qvalue_reward_scaler=self.rew_configs.qvalue_reward_scaler
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


class ChessSFTRewardManager:
    """The reward manager for Chess SFT dataset"""

    def __init__(self, tokenizer, num_examine, rew_configs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.rew_configs = rew_configs


    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        st_time = time.time()

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

            ground_truth_dict = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            # select rm_score
            data_source = data_item.non_tensor_batch["data_source"]
            compute_score_fn = _select_rm_score_fn(data_source)
            
            correct_seq, reward_logs = compute_score_fn(
                    solution_str=sequences_str, 
                    ground_truth_dict=ground_truth_dict,
                    answer_reward=1.0,
                )
            
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
        
        return reward_tensor, normalized_agg_reward_logs