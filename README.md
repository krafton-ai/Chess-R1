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
cd ../../

# LLamaFactory
cd verl/third_party/SkyThought/skythought/train/LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install deepspeed==0.15.4

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

### Download dataset
```
# Download entire dataset
mkdir temp
cd temp
gdown https://drive.google.com/uc?id=1TdaDgXyzmsFNYYIiRi8dxZb-G9H0ctcV
unzip chess_dataset.zip && rm chess_dataset.zip
mv chess_annotation_train.jsonl ../verl/third_party/SkyThought/skythought/train/LLaMA-Factory/data
mv chess_best_move_train.jsonl
mv chess_best_move test.jsonl
mv chess_comparison_train.jsonl
mv chess_comparison test.jsonl
mv chess_mechanics_train.json
mv chess_mechanics_test.json
```

### Zero-shot chess evaluation

### SFT Dataset
```
# Download
# Chess mechanics data
cd verl/third_party/SkyThought/skythought/train/LLaMA-Factory/data
# train
gdown https://drive.google.com/uc?id=10exG7hWuLg0z3TrMGalcUvJnYDunc-qx
# test
gdown https://drive.google.com/uc?id=1BtUx_tqqWHLpqBIeMiWRrlFoXh0C4XFh

# Chess Annotation data (HJ)
# Chess Best move data (HJ)
```

```
# Preprocess
python examples/data_preprocess/chess_sft.py
```

## LLama-Factory
```
# How to use webui in kubeflow
cd verl/third_party/SkyThought/skythought/train/LLaMA-Factory
GRADIO_SHARE=1 llamafactory-cli webui
```


### RL Dataset
```
# If you want to collect your own RL dataset by yourself
wget https://storage.googleapis.com/searchless_chess/data/puzzles.csv
wget https://database.lichess.org/lichess_db_puzzle.csv.zst
unzstd lichess_db_puzzle.csv.zst

python ./searchless_chess/data/extract_lichess.py --save_path ./raw_data --data_path ./lichess_db_puzzle.csv 
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

# San + Fen + Legal moves + Chess rules + Piece Table
python ./examples/data_preprocess/lichess_quiz.py --template_type qwen_instruct_san_fen_legal_rule_table

# Reasoning template + San + Fen + Legal moves + Chess rules + Piece Table
python ./examples/data_preprocess/lichess_quiz.py --template_type qwen_instruct_reasoningtemplate_san_fen_legal_rule_table

# Reasoning template + San + Fen + Legal moves + Chess rules
python ./examples/data_preprocess/lichess_quiz.py --template_type qwen_instruct_reasoningtemplate_san_fen_legal_rule

```

## Supervised Fine-Tuning

GRADIO_SHARE=1 llamafactory-cli webui

## RL Fine-Tuning
```
bash scripts/train_qwen7b_lichess_grpo.sh
```

## Evaluation

### General Reasoning Evaluation (Skythought eval)

```
bash scripts/test_eval.sh
```

### Big Bench Hard (llm-harness)
```
cd verl/third_party
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
conda create -n lm_harness python=3.10.12
conda activate lm_harness
pip install -e .
pip install lm_eval[vllm]
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,dtype="bfloat16" \
    --tasks bbh \
    --batch_size 64 &
```

### GT-Bench
```
hello
```



## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).
* [Chess Dataset](https://github.com/google-deepmind/searchless_chess)
* [Codebase](https://github.com/Jiayi-Pan/TinyZero)
