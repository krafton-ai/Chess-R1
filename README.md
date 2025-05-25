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
cd ../../../

# LLamaFactory
cd verl/third_party/SkyThought/skythought/train/LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install deepspeed==0.15.4
cd ../../../../../../

# JAX
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# utilities
pip install -r requirements.txt
pip install apache_beam


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
bash run_dataset.sh
```

### Zero-shot chess evaluation
```
# Example scripts
bash ./scripts/zeroshot_eval/qwen7b_base/eval_all.sh
```
### SFT Training
```
# Use LLaMA-Factory
# How to use webui in kubeflow
cd verl/third_party/SkyThought/skythought/train/LLaMA-Factory
GRADIO_SHARE=1 llamafactory-cli webui
```

### RL Training

#### RL Dataset
```
# If you want to collect your own RL dataset by yourself
wget https://storage.googleapis.com/searchless_chess/data/puzzles.csv
wget https://database.lichess.org/lichess_db_puzzle.csv.zst
unzstd lichess_db_puzzle.csv.zst

python ./searchless_chess/data/extract_lichess.py --save_path ./raw_data --data_path ./lichess_db_puzzle.csv 

# Change template type according to your needs
python ./data_preprocess/lichess_quiz.py --template_type qwen_instruct_fen_legal_rule_table
```

#### Setup RL feedback model
```
cd searchless_chess
mkdir checkpoints
cd checkpoints
wget https://storage.googleapis.com/searchless_chess/checkpoints/270M.zip
unzip 270M.zip
rm 270M.zip
```


#### RL Fine-Tuning
```
# w/o RL feedback
bash ./scripts/grpo_train/Qwen25_7B_Base_reastemp_fen_legal_rule_norlfeedback.sh
# w/ RL feedback
bash ./scripts/grpo_train/Qwen25_7B_Base_reastemp_fen_legal_rule_rlfeedback.sh
```

## Tensorboard Logging
```
tensorboard --logidr=outputs/{USER_NAME}/{GROUP_NAME}/{EXPERIMENT_NAME}
```


## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).
* [Chess Dataset](https://github.com/google-deepmind/searchless_chess)
* [Codebase](https://github.com/Jiayi-Pan/TinyZero)
