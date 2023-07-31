import pandas as pd
import numpy as np

#Data Visualization
import matplotlib.pyplot as plt

#Text Color
from termcolor import colored

#Train Test Split
from sklearn.model_selection import train_test_split

#Model Evaluation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from mlxtend.plotting import plot_confusion_matrix

#Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model

import wandb
from wandb.keras import WandbCallback

# Login to wandb
wandb.login()
# wandb config
WANDB_CONFIG = {
     'competition': 'AG News Classification Dataset', 
              '_wandb_kernel': 'neuracort'
}

#File Path
TRAIN_FILE_PATH = 'data/train.csv'
TEST_FILE_PATH = 'data/test.csv'

#Load Data
data = pd.read_csv(TRAIN_FILE_PATH)
testdata = pd.read_csv(TEST_FILE_PATH)
#Set Column Names 
data.columns = ['ClassIndex', 'Title', 'Description']
testdata.columns = ['ClassIndex', 'Title', 'Description']

#Combine Title and Description
X_train = data['Title'] + " " + data['Description'] # Combine title and description (better accuracy than using them as separate features)
y_train = data['ClassIndex'].apply(lambda x: x-1).values # Class labels need to begin from 0

x_test = testdata['Title'] + " " + testdata['Description'] # Combine title and description (better accuracy than using them as separate features)
y_test = testdata['ClassIndex'].apply(lambda x: x-1).values # Class labels need to begin from 0

#Max Length of sentences in Train Dataset
maxlen = X_train.map(lambda x: len(x.split())).max()
data.head()
data.shape, testdata.shape
#Checking Value counts to determine class balance
data.ClassIndex.value_counts()
testdata.ClassIndex.value_counts()
#Train Data
data.isnull().sum()
#Test Data
testdata.isnull().sum()
vocab_size = 10000 # arbitrarily chosen
embed_size = 20 # arbitrarily chosen

# Create and Fit tokenizer
tok = Tokenizer(num_words=vocab_size)
tok.fit_on_texts(X_train.values)

# Tokenize data
X_train = tok.texts_to_sequences(X_train)
x_test = tok.texts_to_sequences(x_test)

# Pad data
X_train = pad_sequences(X_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

run = wandb.init(project='ag-news', config= WANDB_CONFIG)

model = Sequential()
model.add(Embedding(vocab_size, embed_size, input_length=maxlen))
model.add(Bidirectional(LSTM(128, return_sequences=True))) 
# Define the layers and their corresponding sizes
layers = [1024, 512, 256, 128, 64]
# Define the dropout rate
dropout_rate = 0.2
for layer_size in layers:
    model.add(Bidirectional(LSTM(layer_size, return_sequences=True)))
    model.add(Dense(layer_size)) #softmax is used as the activation function for multi-class classification problems where class membership is required on more than two class labels.
    model.add(Dropout(dropout_rate))
model.add(GlobalMaxPooling1D()) #Pooling Layer decreases sensitivity to features, thereby creating more generalised data for better test results.
model.add(Dense(4, activation='softmax')) #softmax is used as the activation function for multi-class classification problems where class membership is required on more than two class labels.
model.summary()

callbacks = [
    EarlyStopping(     #EarlyStopping is used to stop at the epoch where val_accuracy does not improve significantly
        monitor='val_accuracy',
        min_delta=1e-4,
        patience=4,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='weights.h5',
        monitor='val_accuracy', 
        mode='max', 
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    WandbCallback()
]

#Compile and Fit Model

model.compile(loss='sparse_categorical_crossentropy', #Sparse Categorical Crossentropy Loss because data is not one-hot encoded
              optimizer='adam', 
              metrics=['accuracy']) 

model.fit(X_train, 
          y_train, 
          batch_size=256, 
          validation_data=(x_test, y_test), 
          epochs=2, 
          callbacks=callbacks)

# Close W&B run
wandb.finish()

model.load_weights('weights.h5')
model.save('model.hdf5')

def modelDemo(news_text):

  #News Labels
  labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']

  test_seq = pad_sequences(tok.texts_to_sequences(news_text), maxlen=maxlen)

  test_preds = [labels[np.argmax(i)] for i in model.predict(test_seq)]

  for news, label in zip(news_text, test_preds):
      # print('{} - {}'.format(news, label))
      print('{} - {}'.format(colored(news, 'yellow'), colored(label, 'blue')))

modelDemo(['New evidence of virus risks from wildlife trade'])
modelDemo(['Coronavirus: Bank pumps £100bn into UK economy to aid recovery'])
modelDemo(['Trump\'s bid to end Obama-era immigration policy ruled unlawful'])
modelDemo(['David Luiz’s future with Arsenal to be decided this week'])
modelDemo(['Indian Economic budget supports the underprivileged sections of society'])
labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']
preds = [np.argmax(i) for i in model.predict(x_test)]
cm  = confusion_matrix(y_test, preds)
plt.figure()
plot_confusion_matrix(cm, figsize=(16,12), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(4), labels, fontsize=12)
plt.yticks(range(4), labels, fontsize=12)
plt.show()

print("Recall of the model is {:.2f}".format(recall_score(y_test, preds, average='micro')))
print("Precision of the model is {:.2f}".format(precision_score(y_test, preds, average='micro')))
print("Accuracy of the model is {:.2f}".format(accuracy_score(y_test, preds)))

from sklearn.metrics import f1_score

# 對測試集的預測結果取最大概率對應的類別標籤
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

# 計算 f1 score
f1 = f1_score(y_test, y_pred, average='weighted')  # 或者 average='micro' 或 'macro'
print("F1 score: {:.2f}".format(f1))