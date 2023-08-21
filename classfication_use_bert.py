import pandas as pd
from sklearn.model_selection import train_test_split
#載入資料
data = pd.read_csv('data/train.csv')
#切割
train_data, test_data = train_test_split(data, textsize = 0.2, random_state = 42)
