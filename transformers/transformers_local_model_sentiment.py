import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

os.environ["HF_MODELS_HOME"] = "E:\\data\\ai_model\\"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 指定本地模型路径
local_model_path = "E:\\data\\ai_model\\"

def pipeline_sentiment_analysis():
    # 加载适用于情感分析任务的预训练模型和分词器
    model_name = "distilbert-base-uncased-distilled-squad"
    local_model_full_path = os.path.join(local_model_path, model_name)

    # 加载本地模型和分词器，确保 local_files_only=True
    tokenizer = AutoTokenizer.from_pretrained(local_model_full_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(local_model_full_path, local_files_only=True)

    # 创建情感分析 pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # 输入文本进行情感分析
    text = "I love using transformers for natural language processing!"
    result = sentiment_pipeline(text)

    # 打印结果
    print(f"Text: {text}")
    print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']}")

if __name__ == '__main__':
    pipeline_sentiment_analysis()
