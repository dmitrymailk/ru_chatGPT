export CUDA_VISIBLE_DEVICES=2
train_log_path=mgpt_$(date +"%d.%m.%Y_%H:%M:%S").log
python mgpt_alpaca_fast.py --micro_batch_size 8 