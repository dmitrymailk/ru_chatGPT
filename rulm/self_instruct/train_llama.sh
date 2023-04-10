export CUDA_VISIBLE_DEVICES=1,4,5
python scripts/train_8bit_adam.py \
 --config-file configs/llama_7b_lora.json \
 --train-file data/train.jsonl --val-file data/val.jsonl \
 --output-dir models/llama_7b_lora