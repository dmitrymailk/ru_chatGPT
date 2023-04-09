python -m \
    fastchat.train.train_mem8 \
    --model_name_or_path /home/dimweb/Desktop/deeppavlov/llama-7b-hf \
    --data_path /home/dimweb/Desktop/deeppavlov/ru_chatGPT/FastChat/playground/data/alpaca-data-conversation.json \
    --bf16 True \
    --output_dir ./checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --tf32 True \