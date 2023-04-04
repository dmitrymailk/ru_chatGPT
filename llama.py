from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch
import time

quantization_config = BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True,
    load_in_8bit=True,
)

MODEL_NAME = "IlyaGusev/llama_7b_ru_turbo_alpaca_lora"

config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map={"": 0},
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
)
# https://github.com/tloen/alpaca-lora/issues/21#issuecomment-1473318920
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

inputs = [
    "Вопрос: Почему трава зеленая?\n\nВыход:",
    "Задание: Сочини длинный рассказ, обязательно упоминая следующие объекты.\nВход: Таня, мяч\nВыход:",
    "Могут ли в природе встретиться в одном месте белый медведь и пингвин? Если нет, то почему?\n\n",
    "Задание: Заполни пропуски в предложении.\nВход: Я пытался ____ от маньяка, но он меня настиг\nВыход:",
    "Как приготовить лазанью?\n\n",
    "Реши уравнение 4x + 5 = 21",
]

model.eval()

# with compile, python 3.11, cuda 11.8 --- 92.65466547012329 seconds ---
# without compile python 3.11, cuda 11.8 --- 79.15323901176453 seconds ---
# with compile, python 3.10, cuda 11.8 --- 79.46766495704651 seconds ---
# without compile python 3.10, cuda 11.6 --- 95.71275281906128 seconds ---
# model = torch.compile(model)

start_time = time.time()

with torch.no_grad():
    for inp in inputs:
        data = tokenizer([inp], return_tensors="pt")
        data = {
            k: v.to(model.device)
            for k, v in data.items()
            if k in ("input_ids", "attention_mask")
        }

        output_ids = model.generate(
            **data,
            num_beams=3,
            max_length=256,
            do_sample=True,
            top_p=0.95,
            top_k=40,
            temperature=1.0,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )[0]
        print(tokenizer.decode(output_ids, skip_special_tokens=True))
        print()
        print("==============================")
        print()

print("--- %s seconds ---" % (time.time() - start_time))
