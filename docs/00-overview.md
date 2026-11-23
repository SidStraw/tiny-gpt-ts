# 00　總覽：一個大語言模型到底在做什麼

這份教學帶你讀懂 `tiny-gpt-ts` 的每一個設計決定。程式語法不會解釋，重點全部放在**概念、演算法，以及「為什麼要這樣做」**。

## 一句話版本

> 大語言模型是一個函數：輸入「目前為止的文字」，輸出「下一個 token 是誰」的機率分布。

就這樣。聊天、寫程式、翻譯，全部都是反覆呼叫這個函數：預測下一個 token、抽一個出來、接到後面、再預測下一個。

這個函數裡面有很多可以調整的數字，稱為**參數**（parameters）。本專案的模型有 229,376 個參數，GPT-2 有 1.24 億個，現代的大模型有數千億個。「訓練」就是調整這些數字，讓函數對真實文字的預測越來越準。

## 需要解決的五個問題

要把上面那句話變成程式，必須回答五個問題。每個問題對應一個章節、一組原始碼：

| 問題 | 答案 | 章節 | 原始碼 |
|------|------|------|--------|
| 參數那麼多，怎麼知道每個該往哪邊調？ | 自動微分（autograd）＋反向傳播 | [01](./01-autograd.md)、[02](./02-tensor.md) | `src/value.ts`、`src/tensor.ts` |
| 文字怎麼變成數字？ | Tokenizer | [03](./03-tokenizer.md) | `src/tokenizer.ts` |
| 函數本身長什麼樣子？ | Transformer（embedding、attention、MLP） | [04](./04-gpt.md) | `src/nn.ts`、`src/gpt.ts` |
| 怎麼調整參數？ | Cross-entropy loss＋AdamW | [05](./05-training.md) | `src/data.ts`、`src/optim.ts`、`src/trainer.ts` |
| 訓練好之後怎麼產生文字？ | 自回歸抽樣 | [06](./06-generation.md) | `src/gpt.ts` 的 `generate` |

最後一章 [07](./07-from-toy-to-real.md) 說明這個玩具和真正的 LLM 差在哪裡。

## 建議的閱讀順序

1. **先玩互動網頁**（`web/dist/index.html`）。七個關卡對應這七份文件，先建立直覺。
2. **讀 `src/value.ts`（約 130 行）搭配第 01 章**。這是整個專案最重要的觀念，其他一切都建立在它上面。
3. **依序讀第 02 到 06 章**，每章對照原始碼。
4. **動手改**：第 07 章最後有練習題。

## 整體資料流

一次訓練步驟，資料是這樣流動的：

```
「春眠不覺曉，處處聞啼鳥。……」
        │  tokenizer.encode                      （第 03 章）
        ▼
  x = [0, 812, 1045, 37, ...]        y = x 往後移一格（正確答案）
        │  查 embedding 表                        （第 04 章）
        ▼
  [T, C] 矩陣：每個 token 一個 C 維向量
        │  N 個 Block（attention + MLP）
        ▼
  [T, C] 矩陣：每個向量已經吸收了前文的資訊
        │  乘上 embedding 表的轉置
        ▼
  logits [T, V]：每個位置對 V 個 token 的評分
        │  cross-entropy（和 y 比較）              （第 05 章）
        ▼
  loss：一個數字，越小表示預測越準
        │  backward()                             （第 01、02 章）
        ▼
  每個參數的梯度：「這個參數變大一點，loss 會變多少」
        │  AdamW
        ▼
  參數更新，loss 下降一點點。重複幾千次。
```

## 符號約定

整份教學使用以下代號，和程式碼裡的命名一致：

| 代號 | 意義 | 本專案預設值 |
|------|------|--------------|
| `V` | vocabSize，字表大小 | 2001（2000 個常用字＋結束符號） |
| `T` | 序列長度（token 數） | 最多 25 |
| `C` | nEmbd，每個 token 的向量維度 | 64 |
| `B` | batch size，一步訓練用幾條序列 | 32 |
| `[T, C]` | 一個 T 列、C 行的矩陣 | |

## 專案結構

```
src/
  value.ts       純量 autograd（觀念版，模型不會用到）
  tensor.ts      矩陣 autograd：13 個運算，各自附 backward
  nn.ts          Linear、Embedding、LayerNorm
  gpt.ts         CausalSelfAttention、MLP、Block、GPT、抽樣與生成
  tokenizer.ts   CharTokenizer、BPETokenizer
  data.ts        Dataset：把文件變成 (x, y) 訓練樣本
  optim.ts       AdamW、gradient clipping、學習率排程
  trainer.ts     trainStep：一步訓練
  checkpoint.ts  存檔與讀檔
  rng.ts         可指定種子的亂數產生器
  cli/           train、generate 兩個指令
tests/           55 個測試，其中最重要的是 gradient check
```

核心（不含 CLI）連同註解約 1,300 行，扣掉註解與空行大約 950 行，沒有任何執行期相依套件。

## 一個貫穿全專案的設計原則：可驗證

從零寫機器學習程式最麻煩的地方是：**寫錯了通常不會當掉，只是結果變差**。梯度少乘一個係數，loss 還是會下降，只是降得慢一點，你根本看不出來。

所以這個專案的每一層都有對應的驗證手段，建議你把它們當成工具記起來：

| 驗證 | 抓得到什麼 | 位置 |
|------|-----------|------|
| Gradient check：autograd 的梯度 vs 數值微分 | 任何 backward 寫錯 | `tests/tensor.test.ts` |
| 初始 loss 應接近 `ln(V)` | 初始化或 loss 計算有問題 | `tests/gpt.test.ts` |
| 因果性測試：改後面的 token，前面的輸出不變 | mask 寫錯、模型偷看答案 | `tests/gpt.test.ts` |
| Overfit 測試：兩條序列要能背到滾瓜爛熟 | 訓練流程任何一環斷掉 | `tests/gpt.test.ts` |
