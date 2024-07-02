import json
import os
import wandb
from datasets import Dataset
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments

os.environ["HF_MODELS_HOME"] = "E:\\data\\ai_model\\"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

aip_key = '98b420c1ea905e27b7885b3d4205832fbef6874f'
# 1.连接 可以写在命令行，也可以写在代码中，只要在代码运行之前运行过即可，这里是代码中的实现
wandb.login(key=aip_key)
# 2.初始化wandb对象，主要用到6的几个参数
runs = wandb.init(
    project="wandb_study",
    # name=f"experiment",
    notes="这是一次test",
    tags=["test", "Test"]
)
# 3.初始化config
# Capture a dictionary of hyperparameters
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}


# 4.找到相应数据并添加，一般的字符串、整形、浮点型直接用字典的形式就可以，图片前面要加wandb.Image()解析成wandb的形式,表格，summary见8和9
# wandb.log({"accuracy": step_acc,
#            "loss": train_loss.item(),
#            'images': wandb.Image(images[0]),
#            })


# 数据准备
def read_json():
    json_data = '''
    [
      {
        "question": "What is the Apollo program?",
        "context": "The Apollo program was the third human spaceflight program carried out by NASA...",
        "answer": "The Apollo program was the third human spaceflight program carried out by NASA"
      }
    ]
    '''
    data = json.loads(json_data)
    # 将数据转换为Dataset对象
    # 转换数据格式
    dataset_dict = {
        "question": [item["question"] for item in data],
        "context": [item["context"] for item in data],
        "answer": [item["answer"] for item in data]
    }

    # 创建Dataset对象
    dataset = Dataset.from_dict(dataset_dict)
    print(dataset)

    return dataset


# 定义数据预处理函数，将输入数据转换为模型可用的格式
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=512,
        truncation=True,
        return_tensors="pt",
        padding="max_length"
    )
    start_positions = []
    end_positions = []
    for i, answer in enumerate(examples["answer"]):
        start_pos = examples["context"][i].find(answer)
        end_pos = start_pos + len(answer)
        start_positions.append(start_pos)
        end_positions.append(end_pos)
    inputs.update({
        "start_positions": start_positions,
        "end_positions": end_positions
    })
    return inputs


# 设置训练参数并初始化Trainer对象
def trainer_training(model):
    processed_dataset = dataset.map(preprocess_function, batched=True)
    training_args = TrainingArguments(
        output_dir='./results',
        run_name='my_experiment',  # 设置一个不同于 output_dir 的 run_name
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps",  # 使用 eval_strategy 替代 evaluation_strategy

    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        eval_dataset=processed_dataset,
    )


if __name__ == '__main__':
    dataset = read_json()

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    trainer = trainer_training(model)

    # 开始训练模型
    trainer.train()

    # 保存训练后的模型
    output_model_dir = "./trained_model"  # 这是一个文件夹，下面有三个文件 config.json model.safetensors training_args.bin
    os.makedirs(output_model_dir, exist_ok=True)
    trainer.save_model(output_model_dir)
