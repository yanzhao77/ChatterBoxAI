import pandas as pd

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader

# 示例数据集
data = [
    {
        "cobol_code": "IDENTIFICATION DIVISION. PROGRAM-ID. HelloWorld. PROCEDURE DIVISION. DISPLAY 'Hello, World!'. STOP RUN.",
        "java_code": "public class HelloWorld { public static void main(String[] args) { System.out.println('Hello, World!'); } }"
    },
    {
        "cobol_code": "IDENTIFICATION DIVISION. PROGRAM-ID. Add. DATA DIVISION. WORKING-STORAGE SECTION. 01 A PIC 9(2) VALUE 10. 01 B PIC 9(2) VALUE 20. PROCEDURE DIVISION. ADD A TO B. DISPLAY B. STOP RUN.",
        "java_code": "public class Add { public static void main(String[] args) { int a = 10; int b = 20; b = a + b; System.out.println(b); } }"
    }
    # 添加更多数据
]


def data_init():
    df = pd.DataFrame(data)
    print(df.head())


class CodeTranslator(nn.Module):
    def __init__(self, config):
        super(CodeTranslator, self).__init__()
        self.encoder = BertModel(config)
        self.decoder = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        decoder_outputs = self.decoder(decoder_input_ids, attention_mask=decoder_attention_mask,
                                       encoder_hidden_states=encoder_outputs[0])
        logits = self.linear(decoder_outputs[0])
        return logits


class CodeTranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cobol_code = self.data[idx]['cobol_code']
        java_code = self.data[idx]['java_code']

        cobol_inputs = self.tokenizer.encode_plus(
            cobol_code, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
        )
        java_inputs = self.tokenizer.encode_plus(
            java_code, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            'input_ids': cobol_inputs['input_ids'].squeeze(),
            'attention_mask': cobol_inputs['attention_mask'].squeeze(),
            'decoder_input_ids': java_inputs['input_ids'].squeeze(),
            'decoder_attention_mask': java_inputs['attention_mask'].squeeze()
        }


def model_train():
    # 加载数据集
    dataset = CodeTranslationDataset(data, tokenizer)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):  # 这里只训练3个epoch作为示例
        for batch in data_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            decoder_input_ids = batch['decoder_input_ids']
            decoder_attention_mask = batch['decoder_attention_mask']

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)

            # 忽略填充部分的损失
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = decoder_input_ids[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")


def model_eval(model):
    model.eval()
    with torch.no_grad():
        test_cobol_code = "IDENTIFICATION DIVISION. PROGRAM-ID. HelloWorld. PROCEDURE DIVISION. DISPLAY 'Hello, World!'. STOP RUN."
        inputs = tokenizer.encode_plus(test_cobol_code, return_tensors='pt')

        decoder_input_ids = torch.zeros((1, 1), dtype=torch.long)  # 初始解码器输入为[CLS] token

        for _ in range(100):  # 最大解码长度
            outputs = model(inputs['input_ids'], inputs['attention_mask'], decoder_input_ids,
                            torch.ones_like(decoder_input_ids))
            next_token_logits = outputs[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.sep_token_id:  # 遇到[SEP] token停止生成
                break

        translated_code = tokenizer.decode(decoder_input_ids.squeeze(), skip_special_tokens=True)
        print("Translated Java Code:", translated_code)


if __name__ == '__main__':
    data_init()
    # 初始化配置和分词器
    config = BertConfig()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 初始化模型
    model = CodeTranslator(config)

    # 训练
    model_train()
    # 推理
    model_eval(model)
