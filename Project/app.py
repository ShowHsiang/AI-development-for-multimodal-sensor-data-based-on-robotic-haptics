import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
)
import warnings
from peft import PeftModel
import time


warnings.filterwarnings("ignore")

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(data_dir):
    X, y = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            label = filename.split('_')[0].split('-')[0]
            df = pd.read_csv(os.path.join(data_dir, filename))
            if df.shape[1] == 65:  # 确保数据有64个传感器值列加一个时间戳列
                data = df.iloc[:, 1:65].values.reshape(-1, 8, 8, 1)
                X.append(data)
                y.append(label)
    return X, y

# Conv3DModel and other utility functions (same as你的现有代码)
class Conv3DModel(nn.Module):
    def __init__(self, num_classes, max_time_steps):
        super(Conv3DModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.fc1 = nn.Linear(128 * (max_time_steps // 8) * 1 * 1, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model(model_path, lora_path=False, quantization='bf16'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        )
    elif quantization == 'int8':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )

    if quantization == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device)
        model.config.attn_implementation = "flash_attention_2"

    for param in model.parameters():
        if param.device.type != device.type:
            raise ValueError(f"Model parameter not on {device.type}: {param.device}")

    config = AutoConfig.from_pretrained(model_path)

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.bfloat16)
        model.to(device)

    return model, tokenizer, device, config

def generate_text(model, tokenizer, device, prompt, args):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        start_time = time.time()
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            **args,
        )
        end_time = time.time()

    total_time = end_time - start_time
    num_tokens = output.size(1)
    tokens_per_second = num_tokens / total_time

    result = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return result, tokens_per_second

def instr_prompt(content):
    final_prompt = "[INST] {} [/INST]".format(content)
    return final_prompt

@app.route('/', methods=['GET', 'POST'])
def index():
    global device
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Load and preprocess data
            X, y = load_data('uploads')
            max_time_steps = 255
            X_padded = []
            for x in X:
                padding = np.zeros((max_time_steps - x.shape[0], 8, 8, 1))
                x_padded = np.vstack((x, padding))
                X_padded.append(x_padded)
            X_padded = np.array(X_padded)
            X_padded = torch.tensor(X_padded, dtype=torch.float32).permute(0, 4, 1, 2, 3)

            conv_model = Conv3DModel(num_classes=136, max_time_steps=max_time_steps)
            conv_model.load_state_dict(torch.load('../dataset/data processing/conv3d_model.pth'))
            conv_model.to(device)
            conv_model.eval()

            y_pred = []
            with torch.no_grad():
                for X_batch in X_padded:
                    X_batch = X_batch.to(device).unsqueeze(0)
                    outputs = conv_model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    y_pred.append(predicted.cpu().numpy()[0])

            df = pd.read_excel('../dataset/data processing/dataset_descriptionEN.xlsx', usecols=['Unnamed: 0', 'Unnamed: 2'])
            df.columns = df.iloc[0]
            df = df[1:]
            df.columns = ['Sample Number', 'Name']
            df['Sample Number'] = df['Sample Number'].apply(lambda x: f"S{x.strip()[:-1]}")
            df['Sample Number'] = df['Sample Number'].str.replace(' ', '')

            fabric_dict = pd.Series(df['Name'].values, index=df['Sample Number']).to_dict()

            def get_fabric_name(label):
                return fabric_dict.get(label, "Unknown Fabric")

            model, tokenizer, device, config = load_model(
                model_path=os.path.join('../ckpts', 'gemma-2-9b-it'),
                quantization='bf16'
            )
            model.eval()

            results = []
            for label in y_pred:
                fabric_name = get_fabric_name(label)
                user_instruction = (
                    "Instruction:\nYou are an experienced expert in the field of fabrics and are able to structurally describe and score different aspects of fabric touch."
                )
                question = f"Detailed describe and rate the touch of {fabric_name} in terms of six dimensions: softness, roughness, smoothness, elasticity, thickness, and temperature."
                prompt = f"{user_instruction}\n Question:\n{question}"
                final_prompt = instr_prompt(content=prompt) + "### Response:\n"
                args = {
                    "temperature": 0.2,
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_new_tokens": 1024,
                    "num_return_sequences": 1
                }
                description, tokens_per_second = generate_text(model, tokenizer, device, final_prompt, args)
                results.append({'fabric': fabric_name, 'description': description, 'tokens_per_second': tokens_per_second})

            return render_template('index.html', results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
