 <svg>
    <foreignObject width="100%" height="100%">
        <style>
            .step {
                color: #CCBBFF;
            }
            .title {
                color: #FF5511;
                font-size:30px;
            }
            .second-title {
                color: #FFCC22;
                font-size:25px;
            }
        </style>
    </foreignObject>
</svg>
 <div class = "title">專案名稱</div>
 
## AG_news classification

<div class = "second-title"> 專案描述 </div>

- ### 新聞分類功能

<div class = "second-title"> 安裝與使用 </div>

### 安裝相依套件

確保你已經安裝以下套件：

- pandas
- numpy
- matplotlib
- termcolor
- scikit-learn
- tensorflow
- mlxtend
- wandb

使用以下指令透過pip安裝相依套件：

```
pip install pandas numpy matplotlib termcolor scikit-learn tensorflow mlxtend wandb
```

<div class = "second-title">下載專案</div>

使用以下命令從GitHub下載專案：

```
git clone https://github.com/elmmaple/AG_news.git
```

<div class = "second-title">執行專案</div>


執行專案的方法：

1. 在終端中cd切至AG_news資料夾底下。

2. 執行Python檔案：

   ```
   python classification.py
   ```
3. 執行時,取得wandb api輸入(需辦理並登入WandB帳戶),
(WandB套件，這是用於跟蹤機器學習實驗的工具，可以記錄模型的準確度、損失函數、學習率等訓練過程中的指標，方便進行實驗結果的比較和分析。)

<div class = "second-title">數據集</div>

- 格式 csv
- 數據來源 https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?fbclid=IwAR2VmOFQDNVGL4wrczn8-cK1rJzESm2OHwFCu10xPlSNIhuEGqefK0De6xY

<div class = "second-title">程式碼結構</div>

<div class = "step">此程式碼用於登入WandB帳戶，使用WandB套件的前提條件，用於將模型訓練過程中的資訊記錄到WandB平台,以下是設置參數,competition比賽名稱及_wandb_kernel指定 W&B 要使用的特定內核名稱，以便在 W&B 平台上識別並運行相關實驗</div>
    
    wandb.login()
    WANDB_CONFIG = {
        'competition': 'AG News Classification Dataset', 
        '_wandb_kernel': 'neuracort'
    }

<div class = "step">接著利用pandas的pd.read_csv載入train和test數據</div>

    pd.read_csv(XXX_FILE_PATH)

<div class = "step">將數據的列名設定為ClassIndex、Title和Description，用於更好地識別數據的特徵再合併在一起作為模型的輸入文本，方便後續模型訓練</div>

    xxx.columns = ['ClassIndex', 'Title', 'Description']
    xxx = xxx['Title'] + " " + xxx['Description']......

<div class = "step">計算文本序列中的最大長度，用於將文本序列填充成相同長度，以便於模型的訓練</div>

    maxlen = X_train.map(lambda x: len(x.split())).max()

<div class = "step">用於探索數據集的分佈和缺失值情況，確保數據集的完整性和一致性</div>

    data.shape, testdata.shape
    data.ClassIndex.value_counts()
    testdata.ClassIndex.value_counts()
    data.isnull().sum()
    testdata.isnull().sum()

<div class = "step">建立Tokenizer進行文本處理。Tokenizer將文本轉換成數字序列，並且將文本序列透過上述maxlen填充成相同長度。</div>

    vocab_size = 10000
    embed_size = 20

    tok = Tokenizer(num_words=vocab_size)
    tok.fit_on_texts(X_train.values)

    X_train = tok.texts_to_sequences(X_train)
    x_test = tok.texts_to_sequences(x_test)

    X_train = pad_sequences(X_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)


<div class = "step">初始化WandB，開始跟蹤訓練過程中的指標</div>

    wandb.init(project='ag-news', config= WANDB_CONFIG)

<div class = "step">建立模型，使用了Embedding層和多層Bidirectional LSTM層，並在每層之後添加了Dropout層。模型的最後一層是全局最大池化層和一個Dense層，用於多類別分類</div>

    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=maxlen))
    model.add(Bidirectional(LSTM(128, return_sequences=True))) 
    layers = [1024, 512, 256, 128, 64]
    dropout_rate = 0.2
    for layer_size in layers:
        model.add(Bidirectional(LSTM(layer_size, return_sequences=True)))
        model.add(Dense(layer_size))
        model.add(Dropout(dropout_rate))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(4, activation='softmax'))
    model.summary()

<div class = "step"> 定義了回調函式，包括EarlyStopping、ModelCheckpoint和WandbCallback。EarlyStopping用於在驗證集的準確度不再改善時停止訓練；ModelCheckpoint用於保存在驗證集上表現最好的模型權重；WandbCallback用於將訓練過程中的資訊記錄到WandB平台</div>

    callbacks = [
    EarlyStopping(
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

<div class = "step"> 編譯並訓練模型，使用了sparse_categorical_crossentropy作為損失函數，用於多類別分類。</div>

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy']) 
    model.fit(X_train, 
            y_train, 
            batch_size=256, 
            validation_data=(x_test, y_test), 
            epochs=2, 
            callbacks=callbacks)

<div class = "step"> 保存模型權重為HDF5文件 </div>

    model.load_weights('weights.h5')
    model.save('model.hdf5')

<div class = "step"> 定義模型測試函式modelDemo，用於輸入新的文本數據並預測其所屬的類別</div>

    def modelDemo
<div class = "step">
    對模型進行測試，輸入了新的文本數據並預測其類別
</div>
    
    modelDemo([.......])

<div class = "step">繪製模型的混淆矩陣，用於評估模型的分類效果</div>
    
    labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']
    preds = [np.argmax(i) for i in model.predict(x_test)]
    cm  = confusion_matrix(y_test, preds)
    plt.figure()
    plot_confusion_matrix(cm, figsize=(16,12), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(4), labels, fontsize=12)
    plt.yticks(range(4), labels, fontsize=12)
    plt.show()

<div class = "step">計算模型的性能指標，包括召回率（recall）、精確率（precision）和準確度（accuracy)</div>

    recall_score
    precision_score
    accuracy_score
    
<div class = "step"> 計算模型的F1分數，用於評估模型的性能。計算了加權F1分數（weighted)</div>

    from sklearn.metrics import f1_score

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    f1 = f1_score(y_test, y_pred, average='weighted')
<div class = "second-title"> 結果</div>

- 準確率和fi.score皆為 0.91