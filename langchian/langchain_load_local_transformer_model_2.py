from typing import Any, List
from pydantic import BaseModel, Field
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import openai
import torch
from langchain.llms import BaseLLM

local_model_path = "E:\\data\\ai_model\\distilbert-base-uncased-distilled-squad"

# 定义 BERT 嵌入模型类，使用 AutoTokenizer 和 AutoModel
class BERTEmbedder:
    def __init__(self, model_path=local_model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# 定义 GPT-3 生成模型类
class GPT3LLM(BaseLLM):
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=150)

    def _generate(self, prompt: str) -> str:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].text.strip()

    def _llm_type(self) -> str:
        return "gpt-3"

    def generate(self, prompt: str) -> str:
        return self._generate(prompt)

from typing import Any

from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel
import openai
import torch
from langchain.llms import Tongyi

local_model_path = "E:\\data\\ai_model\\distilbert-base-uncased-distilled-squad"


# 定义 BERT 嵌入模型类
class BERTEmbedder:
    def __init__(self, model_path=local_model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# 定义 GPT-3 生成模型类
class GPT3LLM(Tongyi):
    def __init__(self, temperature=0.7, max_tokens=150, **data: Any):
        super().__init__(**data)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].text.strip()


class LocalGPT2LLM(Tongyi):
    def __init__(self, model_name=local_model_path, device='cpu'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.device = device

    def generate(self, prompt, max_length=50, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)



# 使用 LangChain 的 LLMChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 定义一个模板，将 BERT 的嵌入作为 GPT-3 的输入
prompt_template = PromptTemplate(
    template="User input embedding: {embedding}\nGenerate response:",
    input_variables=["embedding"]
)

# 实现一个嵌入模型类，用于在 LLMChain 中使用
class EmbeddingsWrapper(Embeddings):
    def __init__(self, embedder):
        self.embedder = embedder

    def embed(self, text):
        return self.embedder.embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text).tolist() for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text).tolist()

# 初始化本地 BERT 嵌入模型和 GPT-3 生成模型
bert_embedder = BERTEmbedder(model_path=local_model_path)
gpt3_llm = GPT3LLM()

# 包装 BERT 嵌入模型
embedding_wrapper = EmbeddingsWrapper(bert_embedder)

# 创建 LLMChain
# llm_chain = LLMChain(
#     llm=gpt3_llm,
#     prompt_template=prompt_template,
# )

# 用户输入
user_input = "The quick brown fox jumps over the lazy dog."

# 处理用户输入并生成输出
embedding = bert_embedder.embed(user_input)
output = gpt3_llm.invoke({"embedding": embedding.tolist()})
print(f"Output: {output}")
