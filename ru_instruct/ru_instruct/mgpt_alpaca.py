import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset


from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template",)

    def __init__(
        self,
    ):
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:",
        }

    def generate_prompt(
        self,
        instruction: str,
        input=None,
        label=None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"

        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def fixed_gpt2_attn(self, query, key, value, attention_mask=None, head_mask=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [],
            value.size(-1) ** 0.5,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )

    # Layer-wise attention scaling
    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        causal_mask = causal_mask > 0
        attn_weights = torch.where(
            causal_mask, attn_weights.to(attn_weights.dtype), mask_value
        )

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


transformers.models.gpt2.modeling_gpt2.GPT2Attention._attn = fixed_gpt2_attn


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./mgpt-instruct",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_run_name: str = "",
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"

    base_model = "ai-forever/mGPT"

    model = transformers.GPT2LMHeadModel.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
    )

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def generate_and_tokenize_prompt(prompt):
        # print(prompt)
        prompt = prompt["prompt"]

        tokenized_prompt = tokenizer(
            prompt,
            max_length=1024,
            truncation=True,
        )
        user_prompt = prompt[: prompt.index("### Assistant:")]
        tokenized_user_prompt = tokenizer(
            user_prompt,
            max_length=1024,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_prompt["labels"] = [-100] * (prompt_len) + tokenized_prompt[
            "input_ids"
        ][prompt_len:]
        return tokenized_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        # target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    datasets_folder = (
        "/home/kosenko/ru_chatGPT/ru_instruct/ru_instruct/sandbox/datasets_processed/"
    )
    data_files = {
        "train": [f"{datasets_folder}{path}" for path in os.listdir(datasets_folder)]
    }
    data = load_dataset(
        "json",
        data_files=data_files,
    )

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # всеравно мы никак не можем адекватно мерить модель
    # тогда зачем тратить на валидацию данные?
    train_data = (
        data["train"]
        .shuffle()
        .map(
            generate_and_tokenize_prompt,
            num_proc=32,
            # batched=True,
        )
    )
    val_data = None

    training_params = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_bnb_8bit",
        evaluation_strategy="no",
        save_strategy="steps",
        eval_steps=None,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        group_by_length=group_by_length,
        report_to="wandb",
        run_name=wandb_run_name,
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_params,
        data_collator=data_collator,
    )

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    model.config.use_cache = False

    model = torch.compile(model)
    trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
