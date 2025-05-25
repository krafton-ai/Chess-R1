# Benchmark Evaluation Tools
This document describes the steps to evaluate LLMs on multiple reasoning benchmarks.

## Requirements 
First create the environment as follows.
```shell
conda create -n eval python==3.10
conda activate eval 
pip install -r requirements.txt
```

For running OpenAI model, export the OpenAI key. 
```shell
export OPENAI_API_KEY={openai_api_key}
```


### Benchmark Evaluations
We provide a wrapper script `eval.py` to conveniently run reasoning benchmarks. We currently support `AIME`, `MATH500`, `GPQADiamond`, and `MMLU`. This script can be used to launch evaluations for multiple benchmarks, then aggregate and log the accuracy for all benchmarks. 

**Note**: The `GPQADiamond` dataset is gated and requires first receiving access at this Huggingface [link](https://huggingface.co/datasets/Idavidrein/gpqa) (which is granted immediately), then logging into your Huggingface account in your terminal session with `huggingface-cli login`. 

**NOTE**: For reproducing `Sky-T1-32B-Preview` results on `AIME` and `GPQADiamond` dataset, pass in temperatures as `0.7`. 

```shell
python eval.py --model NovaSky-AI/Sky-T1-32B-Preview --evals=AIME,GPQADiamond --tp=4 --output_file=results.txt --temperatures 0.7 
```

#### Example Usage
```shell
python eval.py --model Qwen/Qwen2.5-1.5B-Instruct --evals=AIME,MATH500 --tp=4 --output_file=results.txt --temperatures 0.7 
```
    
Example result: `{"AIME": <aime_accuracy>, "MATH500": <math500_accuracy>, "GPQADiamond": <gpqa_diamond_accuracy>}` 
