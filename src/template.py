# CUDA_VISIBLE_DEVICES=1 python template.py
import torch
import os
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, TextStreamer
)
import warnings
from peft import PeftModel
warnings.filterwarnings("ignore")

def load_model(model_path, lora_path=False, quantization='bf16'):
    # Load the tokenizer & set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantization == 'int4':
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2",
        )
    elif quantization == 'int8':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2", 
        )

    if quantization == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device)
        model.config.attn_implementation = "flash_attention_2"

    # 检查模型是否在GPU上
    for param in model.parameters():
        if param.device.type != device.type:
            raise ValueError(f"Model parameter not on {device.type}: {param.device}")

    config = AutoConfig.from_pretrained(model_path)

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.bfloat16)
        model.to(device)  # 再次移动以确保

    return model, tokenizer, device, config

def generate_text(model, tokenizer, device, prompt, args):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            **args,
        )
    
    result = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    question = result.split("### Response:")[0]
    answer = result.split("### Response:\n")[1]

    return question, answer

def turn_weights_to_consolidated_format(model, tokenizer, model_path):
    if hasattr(model, 'module'):
        # The original model is stored in the `module` attribute
        model = model.module
    else:
        # The model is not wrapped with DataParallel, so use it directly
        model = model
    
    # 1.Save the model in consolidated format & name it "consolidated.00.pth"
    # torch.save(model.state_dict(), 'consolidated.00.pth')
    # 2.Save the tokenizer in consolidated format
    # tokenizer.save_pretrained(model_path, save_format="consolidated")

def instr_prompt(content):
    final_prompt = "[INST] {} [/INST]".format(content)
    return final_prompt

def main():
    # 1. Load the model and tokenizer
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_folder = "../ckpts"
    # base_model = "Mistral-7B-Instruct-v0.3"
    # base_model = "gemma-2-9b-it"
    # base_model = "Meta-Llama-3.1-8B-Instruct"
    base_model = "Mistral-Nemo-Instruct-2407"

    # ckpt_folder = "../../../results"
    # base_model = "SusGen_GPT_Mistral_Instruct_v0.3_30k_10epoch_merged"
    model, tokenizer, device, config = load_model(
        model_path=os.path.join(ckpt_folder, base_model),
        # lora_path="../results/SusGen30k-int4-adamw32_Mistral-7B-v0.3/checkpoint-1406",
        quantization='bf16'
    )
    # 2. Set the model to evaluation mode
    model.eval()

    # 3. Define the prompt & generate text
    user_instruction = (
        "Instruction:\nYou are an experienced expert in the field of fabrics and are able to structurally describe and score different aspects of fabric touch."
    )
    question = "Detailed describe and rate the touch of suede in terms of six dimensions: softness, roughness, smoothness, elasticity, thickness, and temperature."
    prompt = f"{user_instruction}\n Question:\n{question}"

    final_prompt = instr_prompt(content=prompt) + "### Response:\n"

    args = {
        # "max_length": 4096,
        "temperature": 0.2,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 40,
        "max_new_tokens": 1024, 
        "num_return_sequences": 1
    }

    question, answer = generate_text(model, tokenizer, device, final_prompt, args)
    print(f"Question:\n{'-' * 10}\n{question.strip()}\n{'=' * 100}")
    print(f"Answer:\n{'-' * 10}\n{answer.strip()}\n{'=' * 100}")

if __name__ == "__main__":
    main()
