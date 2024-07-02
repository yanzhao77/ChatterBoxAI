import os
import torch
os.environ["HF_MODELS_HOME"] = "E:\\data\\ai_model\\"
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set device to CPU
# 将设备设置为CPU
device = torch.device("cpu")

# 加载tokenizer（无设备参数）
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

# 加载模型（无设备参数）
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
model.to(device)
# Encode input text

def tokenize(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Create attention mask (all 1s, since no padding)
    attention_mask = torch.ones_like(input_ids)

    # Generate text using attention mask
    text = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100)

    # Decode generated text
    decoded_text = tokenizer.decode(text[0], skip_special_tokens=True)  # Remove special tokens
    print(decoded_text)

while (True):
    text = input("Enter a sentence: ")
    print(text)
    tokenize(text)