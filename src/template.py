from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_directory = "../ckpts/gemma-2-9b-it"

tokenizer = AutoTokenizer.from_pretrained(model_directory)

model = AutoModelForCausalLM.from_pretrained(model_directory)
model.to("cuda")

prompts = [
    "Translate the following English text to French: 'Hello, how are you?'",
    "Explain the theory of relativity in simple terms."
]

inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

generation_kwargs = {
    "max_length": 128,
    "num_beams": 5,
    "no_repeat_ngram_size": 2,
    "early_stopping": True
}

outputs = model.generate(inputs["input_ids"], **generation_kwargs)

decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
for i, prompt in enumerate(prompts):
    print(f"Prompt: {prompt}")
    print(f"Output: {decoded_outputs[i]}\n")
