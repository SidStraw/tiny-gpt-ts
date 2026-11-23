import { describe, expect, it } from 'vitest'
import { Dataset, parseDocs, splitDocs } from '../src/data.ts'
import { AdamW, clipGradNorm, learningRate } from '../src/optim.ts'
import { RNG } from '../src/rng.ts'
import { Tensor } from '../src/tensor.ts'
import { BPETokenizer, CharTokenizer, tokenizerFromJSON } from '../src/tokenizer.ts'

describe('CharTokenizer', () => {
  const tok = CharTokenizer.fromTexts(['春眠不覺曉，', '處處聞啼鳥。'])

  it('encode 再 decode 會還原', () => {
    expect(tok.decode(tok.encode('春曉，啼鳥。'))).toBe('春曉，啼鳥。')
  })
  it('id 0 保留給結束符號，重複的字只算一次', () => {
    expect(tok.vocabSize).toBe(11 + 1) // 「處」重複，共 11 個不同的字
    expect(tok.encode('春').every((id) => id > 0)).toBe(true)
  })
  it('遇到字表外的字會報錯', () => {
    expect(() => tok.encode('夏')).toThrow()
  })
  it('可序列化', () => {
    expect(tokenizerFromJSON(tok.toJSON()).encode('春眠')).toEqual(tok.encode('春眠'))
  })
})

describe('BPETokenizer', () => {
  it('第一次合併的是最常見的相鄰 pair', () => {
    const tok = BPETokenizer.train(['aaabdaaabac'], 258)
    expect(tok.merges[0]).toEqual([97, 97]) // "aa"
    expect(tok.encode('aaabdaaabac')).toEqual([256, 97, 98, 100, 256, 97, 98, 97, 99])
  })

  it('合併越多，同一段文字的 token 越少', () => {
    const texts = ['the cat sat on the mat', 'the dog sat on the log', 'the cat and the dog']
    const small = BPETokenizer.train(texts, 260)
    const large = BPETokenizer.train(texts, 290)
    expect(large.encode(texts[0]).length).toBeLessThan(small.encode(texts[0]).length)
  })

  it('任何文字都能編碼並還原（含沒看過的字、emoji）', () => {
    const tok = BPETokenizer.train(['春眠不覺曉，處處聞啼鳥。', '夜來風雨聲，花落知多少。'], 300)
    for (const text of ['春眠不覺曉', '完全沒看過的句子 hello 🎉', '']) {
      expect(tok.decode(tok.encode(text))).toBe(text)
    }
  })

  it('中文字一開始是 3 個 byte，訓練後常見字會合併成 1 個 token', () => {
    const tok = BPETokenizer.train(Array(5).fill('春春春春'), 259)
    expect(tok.encode('春')).toHaveLength(1)
    expect(tok.tokenText(tok.encode('春')[0])).toBe('春')
  })

  it('可序列化', () => {
    const tok = BPETokenizer.train(['abababab'], 259)
    const copy = tokenizerFromJSON(JSON.parse(JSON.stringify(tok.toJSON())))
    expect(copy.encode('abab')).toEqual(tok.encode('abab'))
    expect(copy.endToken).toBe(tok.endToken)
  })
})

describe('Dataset', () => {
  const tok = CharTokenizer.fromTexts(['abcdef'])
  const data = new Dataset(['abc', 'abcdef'], tok)

  it('答案 y 是輸入 x 往後移一格，頭尾是結束符號', () => {
    const rng = new RNG(1)
    for (let i = 0; i < 20; i++) {
      const { x, y } = data.sample(rng, 16)
      expect(x[0]).toBe(tok.endToken)
      expect(y.at(-1)).toBe(tok.endToken)
      expect(y.slice(0, -1)).toEqual(x.slice(1))
    }
  })

  it('文件比 blockSize 長時會截成 blockSize', () => {
    const rng = new RNG(2)
    for (let i = 0; i < 20; i++) expect(data.sample(rng, 3).x.length).toBeLessThanOrEqual(3)
    expect(data.maxLength).toBe(7)
  })

  it('parseDocs 與 splitDocs', () => {
    const docs = parseDocs('a\n\n b \nc\nd\n')
    expect(docs).toEqual(['a', 'b', 'c', 'd'])
    const { train, val } = splitDocs(docs, 0.25, new RNG(1))
    expect([train.length, val.length]).toEqual([3, 1])
    expect([...train, ...val].sort()).toEqual(docs)
  })
})

describe('最佳化', () => {
  it('AdamW 能把 (w − 3)² 降到最低點', () => {
    const w = new Tensor(1, 1, [-5])
    const opt = new AdamW([w])
    for (let i = 0; i < 500; i++) {
      opt.zeroGrad()
      w.grad[0] = 2 * (w.data[0] - 3)
      opt.step(0.05)
    }
    expect(w.data[0]).toBeCloseTo(3, 2)
  })

  it('weight decay 只作用在矩陣權重，不作用在 bias', () => {
    const weight = Tensor.filled(2, 2, 1)
    const bias = Tensor.filled(1, 2, 1)
    new AdamW([weight, bias], { weightDecay: 0.5 }).step(0.1) // 梯度為 0，只剩 decay 的效果
    expect(weight.data[0]).toBeCloseTo(0.95, 12)
    expect(bias.data[0]).toBe(1)
  })

  it('clipGradNorm 把梯度長度限制在 maxNorm', () => {
    const p = new Tensor(1, 2)
    p.grad.set([3, 4])
    expect(clipGradNorm([p], 1)).toBeCloseTo(5, 12)
    expect(Math.hypot(p.grad[0], p.grad[1])).toBeCloseTo(1, 12)
  })

  it('學習率：暖身上升、之後沿 cosine 下降到 minRatio', () => {
    const s = { baseLR: 1, warmupSteps: 10, totalSteps: 110, minRatio: 0.1 }
    expect(learningRate(0, s)).toBeCloseTo(0.1, 12)
    expect(learningRate(9, s)).toBeCloseTo(1, 12)
    expect(learningRate(60, s)).toBeCloseTo(0.55, 12)
    expect(learningRate(110, s)).toBeCloseTo(0.1, 12)
  })
})

describe('RNG', () => {
  it('同樣的種子產生同樣的序列', () => {
    const a = new RNG(5), b = new RNG(5)
    expect([a.random(), a.randn(), a.randint(10)]).toEqual([b.random(), b.randn(), b.randint(10)])
  })
  it('multinomial 的抽樣頻率接近給定的機率', () => {
    const rng = new RNG(6)
    const counts = [0, 0, 0]
    for (let i = 0; i < 20000; i++) counts[rng.multinomial([0.2, 0.5, 0.3])]++
    expect(counts[1] / 20000).toBeCloseTo(0.5, 1)
    expect(counts[0] / 20000).toBeCloseTo(0.2, 1)
  })
})
