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

 <div class = "title">專案名稱</div>
 
## AG_news classification

<div class = "second-title"> 專案描述 </div>

- ### 這是一個使用 BERT 模型進行文本分類的範例實作。它展示了如何在 AG News 資料集上訓練並評估基於 BERT 的模型。


<div class = "second-title"> 安裝與使用 </div>

### 安裝相依套件
建議pipenv建置虛擬環境
```
pip install pipenv (先安裝pipenv)
pipenv shell (建置虛擬環境)
pipenv install (已有Pipfile檔案則可直接使用以下指令安裝所需套件)
```

如果上述安裝完成則不需以下分別安裝，總之需確保你已經安裝以下套件：
- torch
- numpy
- pandas
- transformers
- scikit-learn
- tqdm

使用以下指令透過pipenv安裝相依套件：
```
pipenv install pandas numpy torch transformers scikit-learn tqdm
```
<div class = "second-title">下載專案</div>

使用以下命令從GitHub下載專案：

```
git clone https://github.com/elmmaple/AG_news.git
```

<div class = "second-title">執行專案</div>


執行專案的方法：

1. 在終端中cd切至AG_news資料夾底下。

2. 執行Python檔案，這是執行主要程式碼的入口，指令如下

   ```
   python classfication_use_BertForSequenceClassification.py(BertForSequenceClassification版本)
   or 
   python classification_use_BertModel.py(BertModel + 分類器版本)
   ```

<div class = "second-title">數據集</div>

- 格式 csv
- 下載並準備 AG News 數據集。將訓練和測試數據放入 'data' 資料夾中。
- 數據來源 https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?fbclid=IwAR2VmOFQDNVGL4wrczn8-cK1rJzESm2OHwFCu10xPlSNIhuEGqefK0De6xY
----

<div class = "second-title">執行程式流程</div>

    程式將會載入 BERT 模型和分詞器
    定義 TextClassification 模型(如用BertForSequenceClassification則不需額外寫分類器)
    加載訓練和測試數據集
    設定優化器和損失函數
    進行模型訓練
    評估模型在測試集上的表現
    訓練和評估流程
<div class = "second-title">程式碼結構</div>

<div class = "step">資料載入：

    使用 pandas 載入train.csv和test.csv</div>

    pd.read_csv(XXX_FILE_PATH) 

<div class = "step"> 模型建立 </div>
    
    加載預訓練的BERT模型。定義 TextClassification 模型，該模型在 BERT 的基礎上添加了全連接層進行分類
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

<div class = "step"> 數據處理與訓練： </div>

    創建 AGNewsDataset 類別，處理文本數據，進行分詞並準備成模型可接受的格式。
    使用 DataLoader 加載訓練數據，定義優化器和損失函數。
    進行多個 epoch 的訓練，計算損失並進行反向傳播優化

<div class = "step"> 模型評估： </div>

    創建測試數據集並使用 DataLoader 加載。在模型評估模式下，對測試數據進行預測，計算精確度、召回率和 F1 分數等指標。

<div class = "step"> 結果： </div>

    根據測試數據集的預測結果，以下是模型的性能指標：
    精確度（Precision）：{precision}
    召回率（Recall）：{recall}
    測試準確度（Test Accuracy）：{accuracy}
    F1 分數：{f1}