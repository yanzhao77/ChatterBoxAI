from langchain import PromptTemplate, LLMChain
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline

model_path = "E:\\data\\ai_model\\distilbert-base-uncased-distilled-squad"

if torch.cuda.is_available():
    print(torch.cuda.device_count())
    device = torch.device("cuda:0")
    print(device)
else:
    print('没有GPU')
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Ensure the model loads correctly on CPU
if model_path.endswith("4bit"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map={'': device}
    )
elif model_path.endswith("8bit"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={'': device}
    )
else:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(device)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    top_p=1,
    repetition_penalty=1.15,
    device=-1  # Use CPU
)
llama_model = HuggingFacePipeline(pipeline=pipe)

template = '''
#context# 
You are a good helpful, respectful and honest assistant.You are ready for answering human's question and always answer as helpfully as possible, while being safe.
Please ensure that your responses are socially unbiased and positive in nature. 
#question# 
Human: What is a good name for a company that makes {product}?"
'''
prompt = PromptTemplate(
    input_variables=["product"],
    template=template
)
chain = LLMChain(llm=llama_model, prompt=prompt)
print(chain.run("running shoes"))
