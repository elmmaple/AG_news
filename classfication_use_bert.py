import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification
# print(torch.__version__)
# print(torch.cuda.is_available())

#載入資料
data = pd.read_csv('data/train.csv')
#切割
train_data, test_data = train_test_split(data, textsize = 0.2, random_state = 42)
#加載分詞器和模型
tokenizer = BertTokenizerFast('bert-base-uncased')
model = BertForSequenceClassification('bert-base-uncased', num_labels = 4)
device = torch.device("cuda:0" if torch.cuda.is_avaliable() else 'cpu')
model.to(device)
