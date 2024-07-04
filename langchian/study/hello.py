import os

from langchain_community.llms import Tongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class Tongyi_llm():
    def __init__(self):
        api_key = "sk-3f2d7473809f4f0492976b33f3146299"
        os.environ["DASHSCOPE_API_KEY"] = api_key
        self.llm = Tongyi(model_name="qwen2-1.5b-instruct", api_key=api_key)
        self.prompt = ChatPromptTemplate.from_template('{input}')
        # chain_one = LLMChain(llm=llm, prompt=prompt, verbose=False)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def get_completion(self, input_text):
        return self.chain.invoke(input_text)


if __name__ == '__main__':
    model = Tongyi_llm()
    print(model.get_completion("你是通义千问哪个模型，具体型号是多少"))
