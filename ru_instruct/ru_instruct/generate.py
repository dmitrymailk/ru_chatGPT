import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


import torch
import transformers


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


def format_dataset_1(
    instruction="",
    input_data="",
    response="",
    position=1,
):
    if input_data is None:
        input_data = ""

    header = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    header2 = "### Human: Below is an instruction that describes a task. Write a response that appropriately completes the request."
    if position == 1:
        return f"\n\n{header}\n\n{header2}\n\n### Instruction:\n{instruction}\n{input_data}\n### Response:\n### Assistant:\n{response}\n\n"
    elif position == 2:
        return f"\n\n{header}\n\n{header2}\n\n### Instruction:\n{input_data}\n{instruction}\n### Response:\n### Assistant:\n{response}\n\n"


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


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "./mgpt-instruct/checkpoint-2600/",
):
    base_model = "ai-forever/mGPT"
    # base_model = "EleutherAI/pythia-70m"
    # base_model = "./mgpt-instruct/"

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

    model.eval()
    model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = format_dataset_1(
            instruction=instruction,
            input_data=input,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        # output = output[output.rindex("### Assistant:") :]
        return output

    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
    ]:
        # print("Instruction:", instruction)
        print(evaluate(instruction))

        print()


if __name__ == "__main__":
    fire.Fire(main)
