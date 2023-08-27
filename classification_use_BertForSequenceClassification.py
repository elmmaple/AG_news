import torch
import torch.nn as nn
import pandas as pd
import random
import numpy as np
# from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
from tqdm import tqdm  # 引入tqdm庫

# print(torch.__version__)
# print(torch.cuda.is_available())

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
#載入資料
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
#切割
# train_data, test_data = train_test_split(data, textsize = 0.2, random_state = 42)
#加載分詞器和模型
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4) 
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
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
num_epochs = 5
for epoch in range(num_epochs):

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for inputs, labels in train_loader_tqdm:
        inputs = tokenizer(
            inputs,
            max_length = 128,
            truncation = True,
            padding = "max_length",
            return_tensors = "pt"
        )
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        
    # 進行評估
    test_dataset = AGNewsDataset(test_data, tokenizer, max_len = 128)
    test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = False)

    #模型切換到評估模式
    model.eval()
    # torch.no_grad 確保不會因為不必要的計算而增加計算和記憶體負擔，減少內存使用，提高效率
    with torch.no_grad():
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
            
        test_loader_tqdm = tqdm(test_loader, desc="Evaluation")
        for inputs, labels in test_loader_tqdm:
            inputs = tokenizer(
                    inputs,
                    max_length = 128,
                    truncation = True,
                    padding = "max_length",
                    return_tensors="pt"
                )
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
accuracy = correct / total

precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print("Precision", precision)
print("Recall:", recall)
print("Test Accuracy:", accuracy)
print("F1 Score:", f1)