# 07　從玩具到真正的 LLM

讀到這裡，你已經看過一個語言模型的全部零件。這一章回答兩個問題：這個玩具和真正的 LLM 差在哪裡？接下來可以怎麼動手改？

## 1. 相同的部分

先講結論：**演算法是同一個。**

| | tiny-gpt-ts | GPT-2 (small) | 現代大型 LLM |
|---|---|---|---|
| 架構 | Decoder-only Transformer | 相同 | 相同（細節有改良） |
| 訓練目標 | 預測下一個 token | 相同 | 相同（預訓練階段） |
| Loss | Cross-entropy | 相同 | 相同 |
| 梯度 | 反向傳播 | 相同 | 相同 |
| Optimizer | AdamW | Adam | AdamW 及其變體 |
| 生成 | 自回歸抽樣 | 相同 | 相同 |

你在 `src/gpt.ts` 讀到的 `forward`，放大幾百萬倍之後，就是那些模型在做的事。

## 2. 不同的部分

### 2.1 規模

| | tiny-gpt-ts | GPT-2 (small) | 現代大型 LLM |
|---|---|---|---|
| 參數 | 23 萬 | 1.24 億 | 數百億到數千億 |
| 層數 | 2 | 12 | 數十到上百 |
| 向量維度 `C` | 64 | 768 | 數千 |
| Context 長度 | 25 | 1,024 | 數十萬以上 |
| 字表 | 2,001 個字 | 50,257 個 BPE token | 數萬到數十萬 |
| 訓練資料 | 約 33 萬個 token | 數十億個 token | 數兆個 token |
| 硬體 | 一顆 CPU 核心，數十分鐘 | 多張 GPU | 上萬張 GPU，數月 |

模型的能力主要來自規模。我們的模型學到的是「五言詩的格式和常見的字詞搭配」；同樣的演算法在夠大的規模下，為了把下一個 token 預測得更準，被迫學會文法、事實、推理。規模帶來的質變，是這個領域最令人意外的發現。

### 2.2 工程

要達到那種規模，需要大量本專案完全沒有碰到的工程：

- **GPU 與批次運算**：把整個 batch 疊成 3D tensor，矩陣運算交給 GPU 上萬個核心平行處理。我們的 `matmul` 是單執行緒的三層迴圈。
- **混合精度**：用 16 位元浮點數計算，記憶體和時間都減半。
- **分散式訓練**：一個模型切散在上千張 GPU 上，同步梯度。
- **KV cache**：生成時重複利用算過的 key 和 value（第 06 章）。
- **資料處理**：蒐集、清理、去重複、過濾數兆個 token 的文字。

### 2.3 架構上的改良

骨幹沒變，但幾乎每個零件都有更好的版本：

| 本專案（GPT-2 風格） | 現代常見的做法 | 好處 |
|---|---|---|
| Learned absolute position embedding | RoPE（旋轉位置編碼） | 編碼相對位置，較能推廣到長序列 |
| LayerNorm | RMSNorm | 更簡單、更快，效果相當 |
| GELU 的 MLP | SwiGLU | 同樣的參數量下效果稍好 |
| 每個 head 都有獨立的 key、value | Grouped-query attention | 減少 KV cache 的記憶體 |
| 無 dropout | 視情況使用 | 資料量少時防止 overfitting |

這些都是「把同一件事做得更好」，不是不同的想法。

### 2.4 預訓練之後的步驟

我們的訓練只對應 LLM 的第一個階段。要得到一個可以對話的助理，還需要：

1. **預訓練**（pre-training）：在海量文字上預測下一個 token。得到一個「文字接龍」模型。← 本專案在這裡
2. **微調**（fine-tuning）：用對話格式的資料繼續訓練，讓模型學會「看到問題就回答」，而不只是接著寫下去。對話的格式（誰在說話、訊息的邊界）是用特殊 token 標記的，原理和我們的 `<|end|>` 一樣。
3. **對齊**：透過人類回饋、強化學習等方法，讓模型的回答更有幫助、更安全。

第 2、3 階段使用的仍然是同一套機制：算 loss、反向傳播、更新參數。

## 3. 練習題

理解一個系統最好的方法是修改它。以下練習由易到難，每一題都附上提示與驗證方式。

### 入門：只需要改幾行

**1. 換一份資料**
準備任何「一行一份文件」的文字檔（歌詞、人名、程式碼片段），直接訓練：

```bash
pnpm train -- --data 你的檔案.txt
```

觀察：字表大小如何影響初始 loss？文件長度如何影響速度？

**2. 調整超參數**
固定步數，分別改變 `--n-embd`、`--n-layer`、`--lr`。把學習率設成 `0.1` 會發生什麼事？設成 `0.00001` 呢？對照第 05 章的 loss 曲線判讀表。

**3. 把模型弄壞，看測試抓不抓得到**
挑 `src/tensor.ts` 裡的一個 backward，故意改錯（例如把 `softmax` 的 `- dot` 拿掉），執行 `pnpm test`。再試試只跑訓練、不跑測試，看 loss 是否仍然下降。這個實驗會讓你明白為什麼 gradient check 不可或缺。

### 中級：新增一個功能

**4. Top-p 抽樣**
在 `samplingDistribution` 加上 `topP` 選項：把機率由大到小排序，累加到超過 `p` 為止，其餘歸零。
驗證：`topP = 1` 時結果應與不設定相同；`topP` 趨近 0 時應等同 greedy。

**5. Dropout**
新增一個 `dropout(x, p, rng)` 運算：訓練時隨機把比例 `p` 的元素設為 0，其餘乘上 `1/(1−p)`；backward 只讓沒被丟掉的元素通過梯度。加在 attention 和 MLP 的輸出上。
驗證：寫 gradient check（記得讓數值微分的每次 forward 使用相同的 mask）。觀察它是否縮小訓練 loss 與驗證 loss 的差距。

**6. 觀察 attention**
`CausalSelfAttention.lastAttention` 保存了最近一次 forward 的權重。寫一個指令，印出某首詩在各層、各 head 的 attention 矩陣。有沒有哪個 head 專門看「上一句的同一個位置」？（互動網頁的第四關已經做了視覺化，可以對照。）

**7. 儲存驗證 loss 最低的 checkpoint**
目前 `train.ts` 只在最後存檔。改成每次評估時，如果驗證 loss 創新低就存檔。這是最簡單的 early stopping。

### 進階：改動核心

**8. KV cache**
為 `generate` 實作快取。每一層記住先前所有位置的 key 和 value；新 token 進來時只算它自己那一列。
驗證：有無快取的生成結果必須**完全相同**（同樣的種子），然後比較速度。

**9. RoPE**
把 `wpe` 拿掉，改在 attention 裡對 query 和 key 做旋轉。需要新增一個帶 backward 的運算。
驗證：gradient check，以及因果性測試必須繼續通過。

**10. 真正的批次運算**
把 `Tensor` 擴充成支援 3D（`[B, T, C]`），讓一個 batch 只跑一次 forward。你會遇到 padding 與 loss mask 的問題。
驗證：對同一個 batch，新舊兩種做法算出的 loss 與梯度應該一致。

**11. 用 WebGPU 加速 `matmul`**
把矩陣乘法寫成 compute shader。熟悉前端的話，這是最能發揮所長的一題：讓模型在瀏覽器裡用 GPU 訓練。

## 4. 延伸資源

- **Andrej Karpathy〈Let's build GPT: from scratch, in code, spelled out〉**：YouTube 上兩小時的實作影片，用 PyTorch 寫出和本專案相同架構的模型。
- **Karpathy 的 `micrograd`、`nanoGPT`**：本專案 `value.ts` 與 `gpt.ts` 的設計都受到它們啟發，可以對照 Python 版本。
- **〈Attention Is All You Need〉（Vaswani et al., 2017）**：Transformer 的原始論文。讀完本教學之後再看，會發現大部分內容你已經懂了。
- **Sebastian Raschka《Build a Large Language Model (From Scratch)》**：書本形式，涵蓋預訓練之後的微調階段。
- **3Blue1Brown 的神經網路系列影片**：反向傳播與 attention 的視覺化講解。

## 最後

回到一開始的那句話：寫過一次之後，用 LLM 時的判斷力會完全不一樣。

現在你知道了：context window 的上限來自位置編碼和 O(T²) 的 attention；token 是計費單位，因為計算量取決於它；temperature 只是 softmax 之前的一個除法；模型會一本正經地寫出不存在的詩，因為它的工作從頭到尾只是「接下來什麼看起來合理」。

這些不再是需要背的名詞，而是你可以逐行讀懂、親手驗證的程式碼。
