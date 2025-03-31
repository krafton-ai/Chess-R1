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

# spacy_fastlang
pip install spacy_fastlang
pip install cupy-cuda12x
python -m spacy download en_core_web_lg

# install tensorboard (allow host view in kubeflow)
pip install tensorboard
sed -i "s/\"--bind_all\", default=True,/\"--bind_all\",/g" /home/jovyan/conda/chess_llm/lib/python3.10/site-packages/tensorboard/plugins/core/core_plugin.py

# skythought
cd verl/third_party/SkyThought
pip install -e .

# utilities
pip install -r requirements.txt
pip install apache_beam

# JAX
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# tmux
conda install tmux
```

```
source ~/.bashrc
conda activate chess_llm
```

## Chess task

### Dataset
```
cd searchless_chess
mkdir data
cd data
wget https://storage.googleapis.com/searchless_chess/data/puzzles.csv
wget https://database.lichess.org/lichess_db_puzzle.csv.zst
unzstd lichess_db_puzzle.csv.zst
```

```
mkdir train
python extract_lichess.py --save_path ./train/lichess_20k.csv --data_path ./lichess_db_puzzle.csv --data_size 20000

mkdir test
python extract_lichess.py --save_path ./test/lichess_10k.csv --data_path ./puzzles.csv --data_size 10000
```

## Model

```
cd ..
mkdir checkpoints
cd checkpoints
wget https://storage.googleapis.com/searchless_chess/checkpoints/270M.zip
unzip 270M.zip
rm 270M.zip
```

### Preprocessing
```
cd ../..
python ./examples/data_preprocess/lichess_quiz.py --template_type qwen_instruct_san_fen_legal_rule
```

## Countdown task (For debugging)

### Data Preparation
```
# For Base models
python ./examples/data_preprocess/countdown.py

# For Instruct models
python ./examples/data_preprocess/countdown.py --template qwen-instruct
```

## Enjoy

### Training
```
bash scripts/train_qwen7b_lichess_grpo.sh
```

### Evaluation
```
bash scripts/test_eval.sh
```

## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).
* [Chess Dataset](https://github.com/google-deepmind/searchless_chess)
* [Codebase](https://github.com/Jiayi-Pan/TinyZero)