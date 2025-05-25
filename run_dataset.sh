# Download and move dataset files to their appropriate directoroes
mkdir temp
mkdir raw_data
cd temp
gdown https://drive.google.com/uc?id=1TdaDgXyzmsFNYYIiRi8dxZb-G9H0ctcV
unzip chess_dataset.zip && rm chess_dataset.zip

mv chess_annotation_train.jsonl ../verl/third_party/SkyThought/skythought/train/LLaMA-Factory/data
mv chess_best_move_train.jsonl ../verl/third_party/SkyThought/skythought/train/LLaMA-Factory/data
mv chess_comparison_train.jsonl ../verl/third_party/SkyThought/skythought/train/LLaMA-Factory/data
mv chess_mechanics_train.json ../verl/third_party/SkyThought/skythought/train/LLaMA-Factory/data

mv chess_best_move_test.jsonl ../raw_data
mv chess_comparison_test.jsonl ../raw_data
mv chess_mechanics_test.json ../raw_data
mv dm_lichess_10k_processed.csv ../raw_data
mv lichess_db_puzzle_processed.csv ../raw_data

cd ..

# data preprocessing
python data_preprocess/chess_sft.py --data_type chess_comparison
python data_preprocess/chess_sft.py --data_type chess_best_move
python data_preprocess/chess_sft.py --data_type chess_mechanics
python data_preprocess/deepmind_lichess_quiz.py
python data_preprocess/lichess_quiz.py

rm -rf temp

