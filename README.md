# Chess-LLM

## Installation

```
conda create -n chess python=3.10 sqlite==3.42.0
conda activate chess

# install torch
pip install torch==2.4.0

# install vllm
pip3 install vllm==0.5.4 
pip3 install ray[default]

# ignore the warning for the conflict with torch2.4
pip install nvidia-cublas-cu12==12.4.5.8

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation

# install tensorboard (allow host view in kubeflow)
pip install tensorboard
sed -i "s/\"--bind_all\", default=True,/\"--bind_all\",/g" /home/jovyan/conda/chess/lib/python3.10/site-packages/tensorboard/plugins/core/core_plugin.py

# utilities
pip install -r requirements.txt
conda install tmux
```

## Countdown task

**Data Preparation**
```
source ~/.bashrc
conda activate chess
python ./examples/data_preprocess/countdown.py
```

### Run Training

```
bash scripts/test.sh
```

### Run Evaluation

```
cd verl/evaluation
python eval.py --model Qwen/Qwen2.5-1.5B-Instruct --evals=MATH500 --tp=4 --output_file=results.txt --temperatures 0.7 
```

## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).
* [Chess Dataset](https://github.com/google-deepmind/searchless_chess)
* [Codebase](https://github.com/Jiayi-Pan/TinyZero)