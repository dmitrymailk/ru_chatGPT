export CUDA_VISIBLE_DEVICES=3,4
train_log_path=mgpt_$(date +"%d.%m.%Y_%H:%M:%S").log
nohup python mgpt_alpaca.py --micro_batch_size 12 > ./training_logs/train_log_path &