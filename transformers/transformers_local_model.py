from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# 指定本地模型路径
local_model_path = "E:\\data\\ai_model\\distilbert-base-uncased-distilled-squad"

# 加载本地模型和分词器
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained(local_model_path, local_files_only=True)


def print1():
    # 创建问答pipeline
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # 定义上下文和问题
    context = "The Apollo program was the third United States human spaceflight program carried out by NASA."
    question = "What was the Apollo program?"

    # 使用pipeline进行问答
    result = qa_pipeline(question=question, context=context)

    # 打印结果
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")


def hf_pipeline(context, question):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    # 简化处理寻找答案起始和结束位置
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens)
    return answer


def print2():
    # 定义上下文和问题
    context = "The Apollo program was the third United States human spaceflight program carried out by NASA."
    question = "What was the Apollo program?"

    # 直接调用定义的推理函数进行问答
    result = hf_pipeline(context, question)

    print(f"Question: {question}")
    print(f"Answer: {result}")


if __name__ == '__main__':
    print2()
