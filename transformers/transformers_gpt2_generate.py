import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["HF_MODELS_HOME"] = "E:\\data\\ai_model\\"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 对输入文本进行编码
input_ids = tokenizer.encode("你的名字", return_tensors="pt")  # 转换为张量

# 创建注意力掩码（所有为 1，因为没有填充）
attention_mask = torch.ones_like(input_ids)

# 使用注意力掩码生成文本
text = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)

# 对生成文本进行解码
decoded_text = tokenizer.decode(text[0], skip_special_tokens=True)  # 删除特殊令牌
print(decoded_text)
