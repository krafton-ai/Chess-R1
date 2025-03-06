import json
import argparse
import re
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from vllm import LLM, SamplingParams
from tqdm import tqdm
from util.task_handlers import *
from util.model_utils import *
from openai import OpenAI
import concurrent.futures
from functools import partial

# New imports for Hugging Face inference
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def fetch_response_openai(llm, model_name, max_tokens, temp, prompt):
    model_name = model_name.replace("openai/", "")
    if "o1" in model_name:
        # O1 doesn't support system prompt
        # NOTE: might want to implement this inside handler instead 
        for p in prompt:
            p["role"] = "user"
        
        response = llm.chat.completions.create(
            model=model_name,
            messages=prompt,
            n=1,
            temperature=1, # has to be 1
            max_completion_tokens=max_tokens 
        )
    else:
        response = llm.chat.completions.create(
            model=model_name,
            messages=prompt,
            n=1,
            temperature=temp,
            max_tokens=max_tokens
        )
    return response

# Helper function for Hugging Face inference
def hf_generate_responses_batch(model, tokenizer, conversations, max_tokens, temperature, batch_size=4, num_return_sequences=1):
    """Generate responses for a batch of conversations using HuggingFace models with proper padding and attention masks."""
    all_responses = []
    
    # Process conversations in batches
    for i in tqdm(range(0, len(conversations), batch_size)):
        batch = conversations[i:i+batch_size]
        
        # Process the entire batch at once with padding
        tokenized_inputs = tokenizer.apply_chat_template(
            batch,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # Create attention mask based on padding
        attention_mask = (tokenized_inputs != tokenizer.pad_token_id)
        
        # Generate responses with the model
        outputs = model.generate(
            tokenized_inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Reshape outputs if using multiple return sequences
        if num_return_sequences > 1:
            outputs = outputs.view(len(batch), num_return_sequences, -1)
        
        # Process each item in the batch
        batch_responses = []
        for idx, conversation in enumerate(batch):
            prompt_str = tokenizer.apply_chat_template(conversation, tokenize=False)
            prompt_token_ids = tokenizer.encode(prompt_str)
            
            if num_return_sequences > 1:
                # Handle multiple sequences per input
                item_outputs = [outputs[idx, seq_idx] for seq_idx in range(num_return_sequences)]
            else:
                # Single sequence per input
                item_outputs = [outputs[idx]]
            
            response_obj = {
                "outputs": [
                    {
                        "text": tokenizer.decode(output, skip_special_tokens=True).strip(),
                        "token_ids": tokenizer.encode(tokenizer.decode(output, skip_special_tokens=True))
                    }
                    for output in item_outputs
                ],
                "prompt_token_ids": prompt_token_ids
            }
            
            batch_responses.append(response_obj)
        
        all_responses.extend(batch_responses)
        
    return all_responses

def perform_inference_and_check(handler: TaskHandler, temperatures, max_tokens, result_file, llm, system_prompt, args, hf_model=None, hf_tokenizer=None):
    results = handler.load_existing_results(result_file)
    print(f"Loaded {len(results)} existing results.")
    train_data = handler.load_and_filter_dataset(args.start, args.end, split=args.split, source=args.source, \
                                                 filter_difficulty=args.filter_difficulty, args=args)
    remaining_data = handler.process_remaining_data(train_data, results)
    conversations = handler.make_conversations(remaining_data, system_prompt, args.model)

    for temp in temperatures:
        
        if args.model.startswith("openai"):
            fetch_partial = partial(fetch_response_openai, llm, args.model, max_tokens, temp)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
                responses = list(e.map(fetch_partial, conversations))
        else:
            if args.inference_backend == "vllm":
                sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temp)
                responses = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)
            elif args.inference_backend == "huggingface":
                responses = hf_generate_responses_batch(
                    hf_model, hf_tokenizer, conversations, max_tokens, temp, 
                    batch_size=args.batch_size, num_return_sequences=args.n
                )
            else:
                raise ValueError("Unknown inference backend.")

        total_correct = 0 
        total_finish = 0
        with ProcessPoolExecutor(max_workers=32) as executor:
            future_to_task = {}
            token_usages = {}
            for idx, response in enumerate(responses):
                if args.model.startswith("openai"):
                    response_str = response.choices[0].message.content.strip()
                else:
                    response_str = response["outputs"][0]["text"].strip()
                future_to_task[executor.submit(handler.update_results, remaining_data[idx], response_str)] = idx
                
                if args.model.startswith("openai"):
                    token_usages[idx] = response.usage
                else:
                    token_usages[idx] = {
                        "completion_tokens": len(response["outputs"][0]["token_ids"]),
                        "prompt_tokens": len(response["prompt_token_ids"])
                    }

            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Processing Generations"):
                idx = future_to_task[future]
                response_entry = future.result()
                total_correct += response_entry["correctness"]
                total_finish += 1

                problem_key = remaining_data[idx][handler.get_question_key()]
                if problem_key not in results:
                    results[problem_key] = remaining_data[idx]
                    if isinstance(handler, NUMINATaskHandler):
                        results[problem_key]["messages"] = ""
                    results[problem_key]["responses"] = {}
                    results[problem_key]["token_usages"] = {}
                    prompt_text = conversations[idx][1]["content"]
                    results[problem_key]["prompt"] = prompt_text

                results[problem_key]["responses"][str(temp)] = response_entry
                
                if args.model.startswith("openai"):
                    results[problem_key]["token_usages"][str(temp)] = {
                        "completion_tokens": token_usages[idx].completion_tokens,
                        "prompt_tokens": token_usages[idx].prompt_tokens,
                    }
                else:
                    results[problem_key]["token_usages"][str(temp)] = token_usages[idx] 
        
        print(f"Final acc: {total_correct}/{total_finish}")
        acc = round(total_correct / total_finish, 4) if total_finish > 0 else 0
        print(json.dumps({"acc": acc}))

    completion_tokens = [
        results[key].get("token_usages", {}).get(str(temp), {}).get("completion_tokens", 0)
        for key in results for temp in temperatures
    ]
    prompt_tokens = [
        results[key].get("token_usages", {}).get(str(temp), {}).get("prompt_tokens", 0)
        for key in results for temp in temperatures
    ]

    # Token usage summary
    result_dir, result_name = os.path.split(result_file)
    token_usage_dir = os.path.join(result_dir, "token_usage")
    os.makedirs(token_usage_dir, exist_ok=True)
    token_usage_result_file = os.path.join(token_usage_dir, result_name)

    token_dict = {
        "completion_tokens": sum(completion_tokens),
        "prompt_tokens": sum(prompt_tokens),
        "avg_completion_tokens": round(sum(completion_tokens) / len(completion_tokens), 3) if completion_tokens else 0,
        "avg_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens), 3) if prompt_tokens else 0,
    }

    with open(token_usage_result_file, "w") as f:
        json.dump(token_dict, f, indent=4)

    print(f"Token usage saved to {token_usage_result_file}")
    
    with open(result_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

def perform_check(handler: TaskHandler, temperatures, result_file, args):
    results = handler.load_existing_results(result_file)
    print(f"Loaded {len(results)} existing results.")

    train_data = handler.load_and_filter_dataset(args.start, args.end, split=args.split, source=args.source, \
                                                 filter_difficulty=args.filter_difficulty, args=args)
    remaining_data = handler.process_remaining_data(train_data, {})

    tasks = []
    for item in remaining_data:
        problem_key = item[handler.get_question_key()]
        if problem_key in results and "responses" in results[problem_key]:
            for temp in temperatures:
                if str(temp) in results[problem_key]["responses"]:
                    response_entries = results[problem_key]["responses"][str(temp)]
                    for sample_id, response_entry in enumerate(response_entries):
                        if sample_id > (args.n - 1): continue
                        processed = "processed_content" in response_entry
                        tasks.append((item, temp, response_entry["processed_content"] if processed else response_entry["content"], sample_id))

    print(f"Found {len(tasks)} responses requiring reject sampling...")

    total_correct = 0
    total_finish = 0
    correct = { temp: {} for temp in temperatures }
    with ProcessPoolExecutor(max_workers=32) as executor:
        future_to_task = {
            executor.submit(handler.update_results, item, content): (item, temp, sample_id)
            for (item, temp, content, sample_id) in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Processing Reject Sampling"):
            item, temp, sample_id = future_to_task[future]
            new_response_entry = future.result()
            total_correct += new_response_entry["correctness"]
            total_finish += 1
                
            problem_key = item[handler.get_question_key()]
            if problem_key not in correct[temp]:
                correct[temp][problem_key] = False
            if new_response_entry["correctness"]:
                correct[temp][problem_key] = True
            assert problem_key in results and "responses" in results[problem_key] and str(temp) in results[problem_key]["responses"]
            response_entry = results[problem_key]["responses"][str(temp)][sample_id]
            response_entry["correctness"] = new_response_entry["correctness"]
            response_entry["reason"] = new_response_entry["reason"]
            results[problem_key]["responses"][str(temp)][sample_id] = response_entry

    print(f"Final reject-sampling accuracy: {total_correct}/{total_finish}")
    for temp in temperatures:
        temp_correct = sum(correct[temp].values())
        temp_total = len(correct[temp])
        temp_acc = round(temp_correct / temp_total, 4) if temp_total > 0 else 0
        print(f"Temperature {temp} acc: {temp_correct}/{temp_total} ({temp_acc})")

    with open(result_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    
def perform_inference_and_save(handler: TaskHandler, temperatures, max_tokens, result_file, llm, system_prompt, args, hf_model=None, hf_tokenizer=None):
    results = handler.load_existing_results(result_file)
    print(f"Loaded {len(results)} existing results.")
    train_data = handler.load_and_filter_dataset(args.start, args.end, split=args.split, source=args.source, \
                                                 filter_difficulty=args.filter_difficulty, args=args)
    remaining_data = handler.process_remaining_data(train_data, results)
    conversations = handler.make_conversations(remaining_data, system_prompt, args.model)
    
    for temp in temperatures:
        if args.model.startswith("openai"):
            fetch_partial = partial(fetch_response_openai, llm, args.model, max_tokens, temp)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
                responses = list(e.map(fetch_partial, conversations))
        else:
            if args.inference_backend == "vllm":
                sampling_params = SamplingParams(n=args.n, max_tokens=max_tokens, temperature=temp)
                responses = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)
            elif args.inference_backend == "huggingface":
                responses = hf_generate_responses_batch(
                    hf_model, hf_tokenizer, conversations, max_tokens, temp, 
                    batch_size=args.batch_size, num_return_sequences=args.n
                )
            else:
                raise ValueError("Unknown inference backend.")

        completion_tokens = []
        prompt_tokens = []
        for idx, response in enumerate(responses):
            response_entries = []
            token_usages = []
            completion_token = 0
            for sample_idx in range(args.n):
                if args.model.startswith("openai"):
                    content = response.choices[0].message.content.strip()
                else:
                    content = response["outputs"][sample_idx]["text"].strip()
                response_entry = {
                    "content": content,
                    "correctness": None,
                    "reason": None,
                }
                response_entries.append(response_entry)
                if not args.model.startswith("openai"):
                    token_usage = {
                        "completion_tokens": len(response["outputs"][sample_idx]["token_ids"]),
                        "prompt_tokens": len(response["prompt_token_ids"])
                    }
                    token_usages.append(token_usage)
                    completion_token += len(response["outputs"][sample_idx]["token_ids"])
            if not args.model.startswith("openai"):
                completion_token /= args.n
                prompt_token = len(response["prompt_token_ids"])
                prompt_tokens.append(prompt_token)
                completion_tokens.append(completion_token)

            problem_key = remaining_data[idx][handler.get_question_key()]
            if problem_key not in results:
                results[problem_key] = remaining_data[idx]
                if isinstance(handler, NUMINATaskHandler):
                    results[problem_key]["messages"] = ""
                results[problem_key]["responses"] = {}
                results[problem_key]["token_usages"] = {}
                prompt_text = conversations[idx][1]["content"]
                results[problem_key]["prompt"] = prompt_text

            results[problem_key]["responses"][str(temp)] = response_entries
            
            if args.model.startswith("openai"):
                results[problem_key]["token_usages"][str(temp)] = {
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                }
            else:
                results[problem_key]["token_usages"][str(temp)] = token_usages

    result_dir, result_name = os.path.split(result_file)
    token_usage_dir = os.path.join(result_dir, "token_usage")
    os.makedirs(token_usage_dir, exist_ok=True)
    token_usage_result_file = os.path.join(token_usage_dir, result_name)

    token_dict = {
        "completion_tokens": sum(completion_tokens),
        "prompt_tokens": sum(prompt_tokens),
        "avg_completion_tokens": round(sum(completion_tokens) / len(completion_tokens), 3) if completion_tokens else 0,
        "avg_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens), 3) if prompt_tokens else 0,
    }

    with open(token_usage_result_file, "w") as f:
        json.dump(token_dict, f, indent=4)

    print(f"Token usage saved to {token_usage_result_file}")
    
    with open(result_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

def main():
    parser = argparse.ArgumentParser(description="Unified inference and checking for different datasets/tasks.")
    parser.add_argument("--dataset", type=str, required=True, choices=["NUMINA", "APPS", "TACO", "MATH500", "AIME", "GPQADiamond", "MMLU", "MMLUPro", "LiveCodeBench", "GSM8K", "ARC-C"], help="Dataset to process.")
    parser.add_argument("--model", type=str, required=True, default="Qwen/QwQ-32B-Preview", help="The model to run.")
    parser.add_argument("--tp", type=int, default=8, help="Tensor Parallelism Degree")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens for the model.")
    parser.add_argument("--split", type=str, default="train", help="Split to use for apps (e.g., train, test).")
    parser.add_argument("--source", type=str, help="Source for the dataset.")
    parser.add_argument("--start", type=int, default=0, help="Start index.")
    parser.add_argument("--end", type=int, default=-1, help="End index.")
    parser.add_argument("--filter-difficulty", action="store_true", help="Filter difficulty.")
    parser.add_argument("--result-dir", type=str, default="./", help="Result dir to save files.")
    parser.add_argument("--check", action="store_true", help="Perform evaluation checks on generated samples.")
    parser.add_argument("--inference", action="store_true", help="Perform inference.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0], help="Temperature for sampling.")
    parser.add_argument("--math-difficulty-lower-bound", type=int, default=None, help="Lowest difficulty level for math.")
    parser.add_argument("--math-difficulty-upper-bound", type=int, default=None, help="Highest difficulty level for math.")
    parser.add_argument("--n", type=int, default=1, help="Number of samples generated per problem.")
    # New flag: choose backend for non-OpenAI inference
    parser.add_argument("--inference_backend", type=str, choices=["vllm", "huggingface"], default="vllm",
                        help="Inference backend for non-OpenAI models: vllm (default) or huggingface (uses AutoModelForCausalLM).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for Hugging Face inference.")
    args = parser.parse_args()
    
    handler: TaskHandler = TASK_HANDLERS[args.dataset]()
    temperatures = [1] if args.model.startswith("openai/o1") else args.temperatures 
    print(f"Temperature: {temperatures}")
    max_tokens = args.max_tokens
    if temperatures == [0] and args.n > 1:
        args.n = 1
        print("Warning: Temperature 0 does not support multiple samples. Setting n=1.")

    if args.result_dir and not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if args.math_difficulty_lower_bound is not None or args.math_difficulty_upper_bound is not None:
        result_file = os.path.join(args.result_dir, f"{MODEL_TO_NAME[args.model]}_{args.dataset}_{args.split}_{args.source}_{args.start}_{args.end}_{args.math_difficulty_lower_bound}_{args.math_difficulty_upper_bound}.json")
    else:
        result_file = os.path.join(args.result_dir, f"{MODEL_TO_NAME[args.model]}_{args.dataset}_{args.split}_{args.source}_{args.start}_{args.end}.json")

    # Prepare the model object based on the chosen inference backend
    if args.model.startswith("openai"):
        llm = OpenAI()
        hf_model = None
        hf_tokenizer = None
    else:
        if args.inference_backend == "vllm":
            llm = LLM(model=args.model, tensor_parallel_size=args.tp)
            hf_model = None
            hf_tokenizer = None
        elif args.inference_backend == "huggingface":
            hf_tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
            hf_model = AutoModelForCausalLM.from_pretrained(args.model)
            hf_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            llm = None  # not used in this branch
        else:
            raise ValueError("Unknown inference backend.")

    system_prompt = SYSTEM_PROMPT[args.model]

    if args.check:
        if args.math_difficulty_lower_bound is not None or args.math_difficulty_upper_bound is not None:
            converted_file = f"{args.result_dir}/converted_{MODEL_TO_NAME[args.model]}_{args.dataset}_{args.split}_{args.source}_{args.start}_{args.end}_{args.math_difficulty_lower_bound}_{args.math_difficulty_upper_bound}.json"
        else:
            converted_file = f"{args.result_dir}/converted_{MODEL_TO_NAME[args.model]}_{args.dataset}_{args.split}_{args.source}_{args.start}_{args.end}.json"
        if os.path.exists(converted_file):
            result_file = converted_file
        perform_check(handler, temperatures, result_file, args)
        return
    elif args.inference:
        if args.inference_backend == "huggingface":
            perform_inference_and_save(handler, temperatures, max_tokens, result_file, llm, system_prompt, args, hf_model, hf_tokenizer)
        else:
            perform_inference_and_save(handler, temperatures, max_tokens, result_file, llm, system_prompt, args)
        return

    if args.inference_backend == "huggingface":
        perform_inference_and_check(handler, temperatures, max_tokens, result_file, llm, system_prompt, args, hf_model, hf_tokenizer)
    else:
        perform_inference_and_check(handler, temperatures, max_tokens, result_file, llm, system_prompt, args)

if __name__ == "__main__":
    main()
