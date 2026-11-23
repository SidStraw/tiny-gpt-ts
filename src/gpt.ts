// 第三站：GPT 模型本體（decoder-only Transformer）。
//
// 資料流（一條長度 T 的序列）：
//   token ids ─► token embedding + position embedding ─► [T, C]
//             ─► N 個 Block（每個 Block = attention + MLP，各自包在殘差連接裡）
//             ─► LayerNorm ─► 乘上 embedding 表的轉置 ─► logits [T, vocabSize]
// logits 的第 t 列，是模型看完第 0..t 個 token 後，對「第 t+1 個 token 是誰」的評分。
import { Embedding, INIT_STD, LayerNorm, Linear } from './nn.ts'
import { RNG } from './rng.ts'
import {
  add, concatCols, crossEntropy, gelu, maskFuture, matmul, noGrad, scale, sliceCols, softmax, transpose,
} from './tensor.ts'
import type { Tensor } from './tensor.ts'

export interface GPTConfig {
  /** 字表大小：模型認得幾種 token */
  vocabSize: number
  /** context 長度上限：模型一次最多能看幾個 token */
  blockSize: number
  /** Block 疊幾層 */
  nLayer: number
  /** 每層 attention 分成幾個 head */
  nHead: number
  /** 每個 token 的向量維度（必須能被 nHead 整除） */
  nEmbd: number
}

/**
 * Causal self-attention：讓每個 token 去「查詢」它前面的所有 token，把相關資訊搬過來。
 *
 * 每個 token 產生三個向量：
 *   query「我在找什麼」、key「我有什麼」、value「如果你選我，我給你什麼」
 * 第 i 個 token 的 query 和第 j 個 token 的 key 做內積，得到「i 對 j 有多感興趣」的分數；
 * 分數經 softmax 變成權重，再用權重把所有 value 加權平均，就是第 i 個 token 的新資訊。
 *
 * Multi-head：把 C 維切成 nHead 份，每份各自做一次上述流程。
 * 不同 head 可以關注不同的事（例如一個看前一個字、一個看上一句的同位置）。
 */
export class CausalSelfAttention {
  readonly query: Linear
  readonly key: Linear
  readonly value: Linear
  readonly proj: Linear
  readonly nHead: number
  /** 最近一次 forward 的 attention 權重（每個 head 一個 [T, T]），供視覺化使用 */
  lastAttention: Tensor[] = []

  constructor(config: GPTConfig, rng: RNG) {
    const { nEmbd, nHead, nLayer } = config
    if (nEmbd % nHead !== 0) throw new Error('nEmbd 必須能被 nHead 整除')
    this.nHead = nHead
    this.query = new Linear(nEmbd, nEmbd, rng, { bias: false })
    this.key = new Linear(nEmbd, nEmbd, rng, { bias: false })
    this.value = new Linear(nEmbd, nEmbd, rng, { bias: false })
    // 寫回殘差流的最後一層用較小的初始值，讓剛開始訓練時每個 Block 接近「什麼都不做」
    this.proj = new Linear(nEmbd, nEmbd, rng, { std: INIT_STD / Math.sqrt(2 * nLayer) })
  }

  forward(x: Tensor): Tensor {
    const headSize = x.cols / this.nHead
    const q = this.query.forward(x)
    const k = this.key.forward(x)
    const v = this.value.forward(x)

    const heads: Tensor[] = []
    this.lastAttention = []
    for (let h = 0; h < this.nHead; h++) {
      const qh = sliceCols(q, h * headSize, headSize)
      const kh = sliceCols(k, h * headSize, headSize)
      const vh = sliceCols(v, h * headSize, headSize)
      // scores[i][j] = qᵢ · kⱼ；除以 √headSize 讓分數的大小不隨維度膨脹，softmax 才不會太尖銳
      const scores = scale(matmul(qh, transpose(kh)), 1 / Math.sqrt(headSize))
      const attention = softmax(maskFuture(scores))
      this.lastAttention.push(attention)
      heads.push(matmul(attention, vh))
    }
    return this.proj.forward(concatCols(heads))
  }

  parameters(): Tensor[] {
    return [...this.query.parameters(), ...this.key.parameters(), ...this.value.parameters(), ...this.proj.parameters()]
  }
}

/**
 * MLP（feed-forward）：attention 負責「在 token 之間搬資訊」，MLP 負責「消化搬來的資訊」。
 * 它對每個 token 獨立運作：先放大到 4 倍寬、過非線性、再縮回來。
 */
export class MLP {
  readonly fc: Linear
  readonly proj: Linear

  constructor(config: GPTConfig, rng: RNG) {
    this.fc = new Linear(config.nEmbd, 4 * config.nEmbd, rng)
    this.proj = new Linear(4 * config.nEmbd, config.nEmbd, rng, { std: INIT_STD / Math.sqrt(2 * config.nLayer) })
  }

  forward(x: Tensor): Tensor {
    return this.proj.forward(gelu(this.fc.forward(x)))
  }

  parameters(): Tensor[] {
    return [...this.fc.parameters(), ...this.proj.parameters()]
  }
}

/**
 * Transformer Block。注意兩個設計：
 *   1. 殘差連接（x + f(x)）：每個子層只需要學「要在原本的向量上修改什麼」。
 *      梯度也能沿著這條「加法高速公路」直接流回前面的層，深層網路才訓練得動。
 *   2. Pre-norm：LayerNorm 放在子層「之前」，殘差主幹上沒有任何東西擋路。
 */
export class Block {
  readonly ln1: LayerNorm
  readonly attn: CausalSelfAttention
  readonly ln2: LayerNorm
  readonly mlp: MLP

  constructor(config: GPTConfig, rng: RNG) {
    this.ln1 = new LayerNorm(config.nEmbd)
    this.attn = new CausalSelfAttention(config, rng)
    this.ln2 = new LayerNorm(config.nEmbd)
    this.mlp = new MLP(config, rng)
  }

  forward(x: Tensor): Tensor {
    x = add(x, this.attn.forward(this.ln1.forward(x)))
    x = add(x, this.mlp.forward(this.ln2.forward(x)))
    return x
  }

  parameters(): Tensor[] {
    return [...this.ln1.parameters(), ...this.attn.parameters(), ...this.ln2.parameters(), ...this.mlp.parameters()]
  }
}

export class GPT {
  readonly config: GPTConfig
  /** token embedding：每個 token id 一個向量 */
  readonly wte: Embedding
  /** position embedding：每個位置一個向量（attention 本身不知道順序，要靠它告訴模型「誰在第幾個」） */
  readonly wpe: Embedding
  readonly blocks: Block[]
  readonly lnF: LayerNorm

  constructor(config: GPTConfig, rng: RNG = new RNG()) {
    this.config = config
    this.wte = new Embedding(config.vocabSize, config.nEmbd, rng)
    this.wpe = new Embedding(config.blockSize, config.nEmbd, rng)
    this.blocks = Array.from({ length: config.nLayer }, () => new Block(config, rng))
    this.lnF = new LayerNorm(config.nEmbd)
  }

  /** ids（長度 T）→ logits [T, vocabSize] */
  forward(ids: ArrayLike<number>): Tensor {
    const T = ids.length
    if (T > this.config.blockSize) throw new Error(`序列長度 ${T} 超過 blockSize ${this.config.blockSize}`)
    const positions = Array.from({ length: T }, (_, i) => i)
    let x = add(this.wte.forward(ids), this.wpe.forward(positions))
    for (const block of this.blocks) x = block.forward(x)
    x = this.lnF.forward(x)
    // Weight tying：輸出層直接重用 token embedding 表（轉置後相乘）。
    // logits[t][v] = x[t] · wte[v]，也就是「最後的向量和哪個 token 的 embedding 最像」。
    // 輸入和輸出共用同一份「token 的意義」，參數少一大半，通常效果也更好。
    return matmul(x, transpose(this.wte.weight))
  }

  /** 給一條序列和它的正確答案（每個位置的下一個 token），回傳 loss */
  loss(ids: ArrayLike<number>, targets: ArrayLike<number>): Tensor {
    return crossEntropy(this.forward(ids), targets)
  }

  /** 所有可學習的參數（順序固定，存檔與讀檔都依賴這個順序） */
  parameters(): Tensor[] {
    return [
      ...this.wte.parameters(),
      ...this.wpe.parameters(),
      ...this.blocks.flatMap((b) => b.parameters()),
      ...this.lnF.parameters(),
    ]
  }

  parameterCount(): number {
    return this.parameters().reduce((sum, p) => sum + p.size, 0)
  }
}

// ───────────────────────────── 生成 ─────────────────────────────

export interface SamplingOptions {
  /** 溫度：< 1 更保守、> 1 更隨機、0 表示永遠選機率最高的（greedy） */
  temperature?: number
  /** 只從機率最高的 k 個 token 裡抽 */
  topK?: number
}

/**
 * 把一列 logits 變成「要拿來抽樣的機率分布」。
 *   temperature：logits 先除以溫度再做 softmax。溫度低 → 差距被放大 → 分布更尖；溫度高 → 更平。
 *   topK：把第 k 名以後的 token 機率歸零，避免抽到離譜的冷門字。
 */
export function samplingDistribution(logits: ArrayLike<number>, options: SamplingOptions = {}): Float64Array {
  const { temperature = 1, topK } = options
  const n = logits.length
  const probs = new Float64Array(n)

  if (temperature <= 1e-6) {
    let best = 0
    for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i
    probs[best] = 1
    return probs
  }

  let cutoff = -Infinity
  if (topK !== undefined && topK < n) {
    cutoff = Float64Array.from(logits).sort().at(-topK)!
  }
  let max = -Infinity
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i]
  let sum = 0
  for (let i = 0; i < n; i++) {
    probs[i] = logits[i] >= cutoff ? Math.exp((logits[i] - max) / temperature) : 0
    sum += probs[i]
  }
  for (let i = 0; i < n; i++) probs[i] /= sum
  return probs
}

export interface GenerateOptions extends SamplingOptions {
  maxNewTokens: number
  /** 抽到這個 token 就停止 */
  stopToken?: number
  rng: RNG
}

/**
 * 自回歸生成（autoregressive generation）：
 *   預測下一個 token → 抽樣 → 接到序列後面 → 再預測下一個……如此反覆。
 * LLM「一個字一個字吐出來」就是這個迴圈。
 */
export function generate(model: GPT, startIds: number[], options: GenerateOptions): number[] {
  return noGrad(() => {
    const ids = startIds.slice()
    for (let step = 0; step < options.maxNewTokens; step++) {
      const context = ids.slice(-model.config.blockSize) // 超過 context 長度就只看最後一段
      const logits = model.forward(context)
      const lastRow = logits.data.subarray((logits.rows - 1) * logits.cols)
      const next = options.rng.multinomial(samplingDistribution(lastRow, options))
      if (next === options.stopToken) break
      ids.push(next)
    }
    return ids
  })
}
