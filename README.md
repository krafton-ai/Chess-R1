# Chess-LLM

## Installation

```
conda create -n chess_llm python=3.10 sqlite==3.42.0
conda activate chess_llm

# install torch
pip install torch==2.4.0

# install vllm
pip3 install vllm==0.6.3
pip3 install ray[default]

# ignore the warning for the conflict with torch2.4
pip install nvidia-cublas-cu12==12.4.5.8

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation

# install tensorboard (allow host view in kubeflow)
pip install tensorboard
sed -i "s/\"--bind_all\", default=True,/\"--bind_all\",/g" /home/jovyan/conda/chess_llm/lib/python3.10/site-packages/tensorboard/plugins/core/core_plugin.py

# skythought
cd verl/third_party/SkyThought
pip install -e .

# chess dataset
git clone https://github.com/google-deepmind/searchless_chess.git

# utilities
pip install -r requirements.txt
conda install tmux
```

```
source ~/.bashrc
conda activate chess_llm
```

## Chess task

**Data Preparation**
```
cd searchless_chess/data
wget https://storage.googleapis.com/searchless_chess/data/eco_openings.pgn
wget https://storage.googleapis.com/searchless_chess/data/puzzles.csv
mkdir test
cd test
wget https://storage.googleapis.com/searchless_chess/data/test/behavioral_cloning_data.bag
cd ..

mkdir train
cd train
wget https://storage.googleapis.com/searchless_chess/data/train/behavioral_cloning_data.bag
cd ../../..

# For instruct models
python ./examples/data_preprocess/chess_dataset.py --template_type qwen_instruct_with_legal_move
```

**Download Chess model checkpoint**
```
cd searchless_chess/checkpoints
wget https://storage.googleapis.com/searchless_chess/checkpoints/270M.zip
unzip 270M.zip
rm 270M.zip
```


## Countdown task (For debug)

**Data Preparation**
```
# For Base models
python ./examples/data_preprocess/countdown.py

# For Instruct models
python ./examples/data_preprocess/countdown.py --template qwen-instruct
```

### Run Training

```
bash scripts/test_grpo.sh
```

### Run Evaluation

```
skythought evaluate --model Qwen/Qwen2.5-7B-Instruct --task gsm8k --backend vllm --backend-args tensor_parallel_size=1,gpu_memory_utilization=0.8 --sampling-params max_tokens=2048,temperature=0.7,top_p=0.8,top_k=20,repetition_penalty=1.05 --n 1
```

## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).
* [Chess Dataset](https://github.com/google-deepmind/searchless_chess)
* [Codebase](https://github.com/Jiayi-Pan/TinyZero)