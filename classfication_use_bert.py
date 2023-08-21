import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
# print(torch.__version__)
# print(torch.cuda.is_available())

#載入資料
data = pd.read_csv('data/train.csv')
#切割
train_data, test_data = train_test_split(data, textsize = 0.2, random_state = 42)
#加載分詞器和模型
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4) 
device = torch.device("cuda:0" if torch.cuda.is_avaliable() else 'cpu')
model.to(device)

class AGNewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data["Description"].iloc[idx]
        label = self.data["Class Index"].iloc[idx]
        label = label - 1
        return text, label

# 創建DataLoader        
train_dataset = AGNewsDataset(train_data, tokenizer, max_len = 128)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

#設定優化器
optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5)
loss_fn = nn.CrossEntropyLoss()
model.train()

counter = 0

for inputs, labels in train_loader:
    inputs = tokenizer(
        inputs,
        max_length = 128,
        truncation = True,
        padding = "max_length",
        return_tensors = "pt"
    )
    inputs = inputs.to("cuda:0")
    labels = labels.to("cuda:0")
    
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = loss_fn(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    counter += 1
    if counter >= 20:
        break
    