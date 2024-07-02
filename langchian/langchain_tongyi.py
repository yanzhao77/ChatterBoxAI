from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.llms import Tongyi

import os


api_key = "sk-3f2d7473809f4f0492976b33f3146299"

#!pip install langchain langchainhub dashscope

prompt1 = ChatPromptTemplate.from_template('{input}')
os.environ["DASHSCOPE_API_KEY"] = api_key
llm = Tongyi()

chain_one = LLMChain(llm=llm,prompt=prompt1,verbose=False)

print(chain_one.invoke("深圳1998年的情况介绍"))
print(chain_one.invoke("HTML入门"))
print(chain_one.invoke("什么是深度学习"))




