export CUDA_VISIBLE_DEVICES=3,4
python mgpt_alpaca.py \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca' \
    --micro_batch_size 12