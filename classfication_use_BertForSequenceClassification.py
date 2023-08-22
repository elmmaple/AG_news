import torch
import torch.nn as nn
import pandas as pd
# from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
# print(torch.__version__)
# print(torch.cuda.is_available())

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
        inputs = inputs.to("cuda:0")
        labels = labels.to("cuda:0")
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        
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