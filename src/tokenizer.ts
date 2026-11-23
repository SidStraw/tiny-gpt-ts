// 第五站：Tokenizer——把文字切成 token，再把 token 換成整數 id。
// 模型只看得懂數字；tokenizer 決定了「模型眼中的最小單位」是什麼。

export interface Tokenizer {
  readonly vocabSize: number
  /** 文件分隔符號：同時代表「開始」與「結束」 */
  readonly endToken: number
  encode(text: string): number[]
  decode(ids: ArrayLike<number>): string
  /** 單一 token 的文字（視覺化用） */
  tokenText(id: number): string
  toJSON(): TokenizerJSON
}

export type TokenizerJSON =
  | { type: 'char'; chars: string[] }
  | { type: 'bpe'; merges: [number, number][] }

export const END_TEXT = '<|end|>'

/**
 * 字元級 tokenizer：一個字就是一個 token。
 * 優點是簡單、字表小；缺點是序列長，而且模型得自己學會「哪些字常常黏在一起」。
 * 對中文古詩來說，一字一 token 剛好非常自然。
 */
export class CharTokenizer implements Tokenizer {
  readonly endToken = 0
  readonly chars: string[]
  private readonly ids = new Map<string, number>()

  /** chars[i] 的 id 是 i + 1（id 0 保留給結束符號） */
  constructor(chars: string[]) {
    this.chars = chars
    chars.forEach((ch, i) => this.ids.set(ch, i + 1))
  }

  static fromTexts(texts: string[]): CharTokenizer {
    const set = new Set<string>()
    for (const t of texts) for (const ch of t) set.add(ch)
    return new CharTokenizer([...set].sort())
  }

  get vocabSize(): number {
    return this.chars.length + 1
  }

  encode(text: string): number[] {
    return Array.from(text, (ch) => {
      const id = this.ids.get(ch)
      if (id === undefined) throw new Error(`字表裡沒有「${ch}」這個字`)
      return id
    })
  }

  decode(ids: ArrayLike<number>): string {
    return Array.from(ids, (id) => (id === this.endToken ? '' : this.chars[id - 1])).join('')
  }

  tokenText(id: number): string {
    return id === this.endToken ? END_TEXT : this.chars[id - 1]
  }

  toJSON(): TokenizerJSON {
    return { type: 'char', chars: this.chars }
  }
}

/**
 * Byte Pair Encoding（BPE）：GPT 系列實際使用的做法。
 *
 * 訓練：
 *   1. 把文字轉成 UTF-8 bytes，一開始每個 byte（0–255）就是一個 token。
 *   2. 找出「最常相鄰出現的一對 token」，把它們合併成一個新 token（id 從 256 開始編）。
 *   3. 重複步驟 2，直到字表長到想要的大小。
 * 常見的字、詞會被合併成單一 token，罕見的字則拆成幾個碎片。
 * 因此字表大小固定、卻能表示任何文字（最差也能退回一個個 byte）。
 *
 * 編碼：依照訓練時學到的合併順序，對新文字重做同樣的合併。
 */
export class BPETokenizer implements Tokenizer {
  /** merges[i] = [a, b] 表示 token a 和 b 合併成 token 256 + i */
  readonly merges: [number, number][]
  readonly endToken: number
  /** 每個 token 對應的 bytes */
  private readonly bytes: Uint8Array[] = []
  /** 一對 token → 合併後的 id。id 越小表示越早學到，編碼時優先合併。 */
  private readonly rank = new Map<number, number>()

  constructor(merges: [number, number][]) {
    this.merges = merges
    for (let i = 0; i < 256; i++) this.bytes.push(Uint8Array.of(i))
    merges.forEach(([a, b], i) => {
      this.bytes.push(Uint8Array.of(...this.bytes[a], ...this.bytes[b]))
      this.rank.set(pairKey(a, b), 256 + i)
    })
    this.endToken = 256 + merges.length
  }

  static train(texts: string[], vocabSize: number): BPETokenizer {
    const numMerges = vocabSize - 257 // 256 個 byte + 1 個結束符號
    if (numMerges < 0) throw new Error('BPE 的 vocabSize 至少要 257')
    const encoder = new TextEncoder()
    let docs = texts.map((t) => Array.from(encoder.encode(t)))
    const merges: [number, number][] = []

    for (let i = 0; i < numMerges; i++) {
      // 統計每一對相鄰 token 出現幾次（不跨文件）
      const counts = new Map<number, number>()
      for (const doc of docs) {
        for (let j = 0; j + 1 < doc.length; j++) {
          const key = pairKey(doc[j], doc[j + 1])
          counts.set(key, (counts.get(key) ?? 0) + 1)
        }
      }
      let bestKey = -1, bestCount = 1 // 至少要出現 2 次才值得合併
      for (const [key, count] of counts) {
        if (count > bestCount) [bestKey, bestCount] = [key, count]
      }
      if (bestKey < 0) break
      const pair: [number, number] = [Math.floor(bestKey / PAIR_BASE), bestKey % PAIR_BASE]
      merges.push(pair)
      docs = docs.map((doc) => mergePair(doc, pair[0], pair[1], 256 + i))
    }
    return new BPETokenizer(merges)
  }

  get vocabSize(): number {
    return this.endToken + 1
  }

  encode(text: string): number[] {
    let ids = Array.from(new TextEncoder().encode(text))
    while (ids.length >= 2) {
      // 在目前所有相鄰的 pair 中，找「最早被學到」的那個合併規則
      let best = Infinity, bestIndex = -1
      for (let j = 0; j + 1 < ids.length; j++) {
        const merged = this.rank.get(pairKey(ids[j], ids[j + 1]))
        if (merged !== undefined && merged < best) [best, bestIndex] = [merged, j]
      }
      if (bestIndex < 0) break
      ids = mergePair(ids, ids[bestIndex], ids[bestIndex + 1], best)
    }
    return ids
  }

  decode(ids: ArrayLike<number>): string {
    const out: number[] = []
    for (let i = 0; i < ids.length; i++) {
      if (ids[i] !== this.endToken) out.push(...this.bytes[ids[i]])
    }
    return new TextDecoder().decode(Uint8Array.from(out))
  }

  tokenText(id: number): string {
    if (id === this.endToken) return END_TEXT
    return new TextDecoder().decode(this.bytes[id])
  }

  toJSON(): TokenizerJSON {
    return { type: 'bpe', merges: this.merges }
  }
}

const PAIR_BASE = 1 << 20
function pairKey(a: number, b: number): number {
  return a * PAIR_BASE + b
}

/** 把序列中所有相鄰的 (a, b) 換成 merged */
function mergePair(ids: number[], a: number, b: number, merged: number): number[] {
  const out: number[] = []
  for (let j = 0; j < ids.length; j++) {
    if (j + 1 < ids.length && ids[j] === a && ids[j + 1] === b) {
      out.push(merged)
      j++
    } else {
      out.push(ids[j])
    }
  }
  return out
}

export function tokenizerFromJSON(json: TokenizerJSON): Tokenizer {
  return json.type === 'char' ? new CharTokenizer(json.chars) : new BPETokenizer(json.merges)
}
