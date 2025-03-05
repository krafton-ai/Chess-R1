# Chess-LLM

## Installation

```
conda create -n chess python=3.9
conda activate chess
# install torch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.5.4
pip3 install ray[default]

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# utilities
pip install wandb IPython matplotlib
conda install tmux
```

## Countdown task

**Data Preparation**
```
conda activate chess
python ./examples/data_preprocess/countdown.py
python ./examples/data_preprocess/countdown.py --template_type=qwen-instruct
```

### Run Training

#### Single GPU (<3B GPU)

Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning.
```
bash ./scripts/train_tiny_zero.sh
```

#### Multi GPU (<3B GPU)

In this case, the base model is able to develop sophisticated reasoning skills.
```
bash scripts/exps/train_tiny_zero_grpo_qwen7b_instruct.sh
```

## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).
* [Chess Dataset](https://github.com/google-deepmind/searchless_chess)
* [Codebase](https://github.com/Jiayi-Pan/TinyZero)