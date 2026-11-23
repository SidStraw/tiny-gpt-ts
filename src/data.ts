// 資料集：把文件變成訓練樣本。
//
// 語言模型的訓練資料不需要人工標註——「正確答案」就是文字本身往後移一格：
//   輸入 x：<end> 春 眠 不 覺 曉
//   答案 y： 春  眠 不 覺 曉 <end>
// 一條長度 T 的序列，同時提供了 T 個「看到這些 → 下一個是誰」的練習題。
import type { RNG } from './rng.ts'
import type { Tokenizer } from './tokenizer.ts'

export interface Example {
  x: number[]
  y: number[]
}

/** 一行一份文件，忽略空行 */
export function parseDocs(text: string): string[] {
  return text.split('\n').map((line) => line.trim()).filter((line) => line.length > 0)
}

/** 洗牌後切成訓練集與驗證集。驗證集不參與訓練，用來檢查模型是「學會」還是「背起來」。 */
export function splitDocs(docs: string[], valFraction: number, rng: RNG): { train: string[]; val: string[] } {
  const shuffled = rng.shuffle(docs.slice())
  const valCount = Math.max(1, Math.floor(shuffled.length * valFraction))
  return { train: shuffled.slice(valCount), val: shuffled.slice(0, valCount) }
}

export class Dataset {
  /** 每份文件編碼後的 token，前後各加一個結束符號 */
  readonly docs: number[][]

  constructor(texts: string[], tokenizer: Tokenizer) {
    this.docs = texts.map((t) => [tokenizer.endToken, ...tokenizer.encode(t), tokenizer.endToken])
  }

  /** 最長的文件需要多大的 context（輸入長度 = token 數 − 1） */
  get maxLength(): number {
    return this.docs.reduce((max, d) => Math.max(max, d.length - 1), 0)
  }

  /** 隨機抽一份文件；如果比 blockSize 長，就隨機截一段 */
  sample(rng: RNG, blockSize: number): Example {
    const doc = this.docs[rng.randint(this.docs.length)]
    const length = Math.min(blockSize, doc.length - 1)
    const start = rng.randint(doc.length - length)
    return { x: doc.slice(start, start + length), y: doc.slice(start + 1, start + length + 1) }
  }

  batch(rng: RNG, batchSize: number, blockSize: number): Example[] {
    return Array.from({ length: batchSize }, () => this.sample(rng, blockSize))
  }
}
