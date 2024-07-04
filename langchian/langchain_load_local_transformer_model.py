from langchain.chains.llm import LLMChain
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# Qwen2-7B-Instruct
# Llama3-8B-Chinese-Chat
# 使用GPT-2模型路径

# tiny-gpt2 gpt2
local_model_path = r"E:\data\ai_model\distilbert-base-uncased-distilled-squad"

# 加载 GPT-2 模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# 创建 pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=512,
    do_sample=True,
    top_k=30,
    num_return_sequences=1
)

# 使用 HuggingFacePipeline 包装 pipeline
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

# 定义系统提示和模板
system_prompt = ""
instruction = ""
template = '''
#context# 
你是一个智能助手，对于用户提出的问题，你可以简洁又专业的给出回答。当用户对你打招呼，或者问你的身份问题，或者让你做自我介绍时，都要回答如下：我是来自xx的智能助手，我是由xx工程师开发的. 
#question# 
Human: 回答如下问题:\n\n {input}"
'''

# 使用 ChatPromptTemplate 创建 prompt
# prompt = ChatPromptTemplate.from_template(template)
prompt = PromptTemplate(template=template, input_variables=["text"])
# 创建 LLMChain
llm_chain = prompt | llm | StrOutputParser()

# 用户输入
text = "你是谁，你来自哪里。告诉我太阳为什么从东边升起！"

# 生成输出
output = llm_chain.invoke(text)
print(output)

