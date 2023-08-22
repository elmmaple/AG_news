import torch
import torch.nn as nn
import pandas as pd
# from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
from typing import Optional

# print(torch.__version__)
# print(torch.cuda.is_available())

#載入資料
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
#加載分詞器和模型
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased') 
# device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# model.to(device)

class TextClassfication(nn.Module):
    def __init__(self, model, num_labels):
        super(TextClassfication, self).__init__()
        self.model = model
        #减少神经元之间的相互依赖關係，避免模型隊訓練數據的過擬合，提高模型在未見過的數據上的泛化能力。
        self.dropout = nn.Dropout(0.1)
        #線性層 Bert模型特徵映射分類類別數量
        self.classifier = nn.Linear(in_features = 768, out_features = num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
        
model = TextClassfication(bert_model, num_labels=4)

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
    # inputs = inputs.to("cuda:0")
    # labels = labels.to("cuda:0")
    
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    counter += 1
    if counter >= 20:
        break
    
# 進行評估
test_dataset = AGNewsDataset(test_data, tokenizer, max_len = 128)
test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = False)

counter = 0
#模型切換到評估模式
model.eval()
# torch.no_grad 確保不會因為不必要的計算而增加計算和記憶體負擔，減少內存使用，提高效率
with torch.no_grad():
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    for inputs, labels in test_loader:
        inputs = tokenizer(
                inputs,
                max_length = 128,
                truncation = True,
                padding = "max_length",
                return_tensors="pt"
            )
        # inputs = inputs.to("cuda:0")
        # labels = labels.to("cuda:0")
        outputs = model(**inputs)
        predictions = torch.argmax(outputs, dim=1)
        
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        counter += 1
        if counter >= 10:
            break
        
accuracy = correct / total
precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print("Precision", precision)
print("Recall:", recall)
print("Test Accuracy:", accuracy)
print("F1 Score:", f1)