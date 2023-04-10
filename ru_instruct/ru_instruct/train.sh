export CUDA_VISIBLE_DEVICES=3,4
python mgpt_alpaca.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca' \
    --micro_batch_size 64