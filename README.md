# tiny-gpt-ts

從零開始、不靠任何機器學習函式庫，用 TypeScript 寫出的迷你 GPT。它會寫五言詩。

整個專案是為了**學習**而寫的：autograd、矩陣運算、Transformer、tokenizer、optimizer 全部自己實作，每個設計決定都有註解與文件說明。核心程式碼扣掉註解大約 950 行，沒有任何執行期相依套件。

```
$ pnpm generate -- --prompt 春
春來一點淚，春早始知音。今日有心夕，歸家見故鄉。
春谷獨歸客，客鄉情在岐。不知還得見，誰共故山西。
```

## 三種學習方式

| | 內容 | 位置 |
|---|---|---|
| **互動網頁** | 七個關卡，每關有可操作的示範與一道試煉。頁面裡跑的就是 `src/` 的程式碼 | [線上版](https://sidstraw.github.io/tiny-gpt-ts/)，或直接用瀏覽器開啟 `web/dist/index.html` |
| **教學文件** | 八份文件，說明所有概念、演算法與設計理由 | [`docs/00-overview.md`](./docs/00-overview.md) |
| **原始碼** | 每個檔案開頭說明它在整體中的角色，每個運算附推導 | `src/` |

建議順序：先玩網頁建立直覺，再讀文件對照原始碼，最後做 [`docs/07`](./docs/07-from-toy-to-real.md) 的練習題。

## 快速開始

需要 Node.js 22.18 以上（直接執行 TypeScript）與 pnpm。

```bash
pnpm install
pnpm test                          # 55 個測試，含所有梯度的數值驗證
pnpm prepare-data                  # 下載《全唐詩》並整理成 data/poems.txt
pnpm train                         # 訓練 3,000 步，存到 checkpoints/model.json
pnpm generate -- --prompt 明月      # 寫詩
pnpm build:web                     # 重新產生 web/dist/index.html
```

### 訓練選項

```bash
pnpm train -- --steps 1000 --n-embd 32 --n-layer 1 --batch-size 8   # 小模型，一分鐘內跑完
pnpm train -- --data 你的檔案.txt                                    # 任何「一行一份文件」的文字檔
pnpm train -- --tokenizer bpe --vocab-size 600                      # 改用 BPE tokenizer
```

| 選項 | 預設 | 說明 |
|---|---|---|
| `--data` | `data/poems.txt` | 訓練資料 |
| `--steps` | `3000` | 訓練步數 |
| `--batch-size` | `32` | 每步幾條序列 |
| `--n-layer` / `--n-head` / `--n-embd` | `2` / `4` / `64` | 模型大小 |
| `--lr` | `0.003` | 最高學習率 |
| `--tokenizer` | `char` | `char` 或 `bpe` |
| `--out` | `checkpoints/model.json` | 存檔位置 |

### 生成選項

| 選項 | 預設 | 說明 |
|---|---|---|
| `--prompt` | 空 | 詩的開頭 |
| `--count` | `5` | 寫幾首 |
| `--temperature` | `0.8` | 越低越保守，`0` 為永遠選機率最高的字 |
| `--top-k` | `40` | 只從前 k 名裡抽 |
| `--seed` | 隨機 | 指定後結果可重現 |

## 訓練結果

預設設定（23 萬個參數）在 13,527 首詩上訓練 3,000 步，M4 Pro 單核心約 23 分鐘：

| 步數 | 訓練 loss | 驗證 loss | 當時寫出的詩 |
|------|-----------|-----------|--------------|
| 0 | 7.59 | 7.58 | 騷延求剪僧翔信信陸夷康迥酒傳流暎泊立簪雁庾冷銀崇禍 |
| 100 | 5.50 | 5.31 | 山道江歸，，歸去一相逢。西相知，，，何在自一寒。 |
| 500 | 4.61 | 4.62 | 南雲寒已盡，不得秋深盡。唯思如何處，難見見山風。 |
| 1000 | 4.32 | 4.50 | 野色深秋夜，遙時入雨時。誰言與高手，歸意一時愁。 |
| 2000 | 4.02 | 4.36 | 石轉三千里，山閑一四鄰。不同無有志，何用老空書。 |
| 2999 | 3.94 | 4.28 | 夜高高月裏，春色下朝煙。此去一離恨，無心正自傷。 |

模型學會了五言詩的格式與常見的字詞搭配。它寫的詩看起來像唐詩，但內容是編的；這個規模的模型還談不上理解語意。詳細的判讀見 [`docs/05-training.md`](./docs/05-training.md)。

## 專案結構

```
src/
  value.ts       純量 autograd（觀念版，約 120 行，建議第一個讀）
  tensor.ts      矩陣 autograd：13 個運算，各自附 backward
  nn.ts          Linear、Embedding、LayerNorm
  gpt.ts         CausalSelfAttention、MLP、Block、GPT、抽樣與生成
  tokenizer.ts   CharTokenizer、BPETokenizer
  data.ts        Dataset：把文件變成 (x, y) 訓練樣本
  optim.ts       AdamW、gradient clipping、學習率排程
  trainer.ts     trainStep：一步訓練
  checkpoint.ts  存檔與讀檔
  rng.ts         可指定種子的亂數產生器
  cli/           train、generate 指令
tests/           gradient check、因果性、overfit 等 55 個測試
docs/            教學文件
web/src/         互動網頁的原始檔；web/dist/ 為打包結果
scripts/         資料準備、網頁打包
```

## 資料來源

訓練資料取自 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（MIT 授權）的《全唐詩》。`pnpm prepare-data` 挑出五言絕句與五言律詩（律詩每兩聯拆成一首），只保留由最常見 2,000 字組成的作品，共 15,030 首。

## 致謝

設計深受 Andrej Karpathy 的 [micrograd](https://github.com/karpathy/micrograd) 與 [nanoGPT](https://github.com/karpathy/nanoGPT) 啟發。
