import { describe, expect, it } from 'vitest'
import { loadCheckpoint, saveCheckpoint } from '../src/checkpoint.ts'
import { generate, GPT, samplingDistribution } from '../src/gpt.ts'
import type { GPTConfig } from '../src/gpt.ts'
import { AdamW } from '../src/optim.ts'
import { RNG } from '../src/rng.ts'
import { noGrad } from '../src/tensor.ts'
import { CharTokenizer } from '../src/tokenizer.ts'
import { batchLoss, trainStep } from '../src/trainer.ts'

const config: GPTConfig = { vocabSize: 11, blockSize: 8, nLayer: 2, nHead: 2, nEmbd: 8 }

/** 把初始值放大，讓每條路徑的梯度都夠明顯（預設的 0.02 太小，數值微分會被捨入誤差淹沒） */
function testModel(): GPT {
  const model = new GPT(config, new RNG(3))
  const rng = new RNG(4)
  for (const p of model.parameters()) for (let i = 0; i < p.size; i++) p.data[i] += rng.randn() * 0.3
  return model
}

describe('GPT', () => {
  it('輸出形狀為 [T, vocabSize]', () => {
    const logits = new GPT(config).forward([1, 2, 3, 4, 5])
    expect([logits.rows, logits.cols]).toEqual([5, 11])
  })

  it('剛初始化時，loss 接近亂猜的 ln(vocabSize)', () => {
    const loss = new GPT(config).loss([1, 2, 3, 4], [2, 3, 4, 5]).item()
    expect(loss).toBeCloseTo(Math.log(11), 1)
  })

  it('因果性：改變後面的 token，不會影響前面位置的輸出', () => {
    const model = testModel()
    const a = noGrad(() => model.forward([1, 2, 3, 4, 5, 6]))
    const b = noGrad(() => model.forward([1, 2, 3, 9, 9, 9]))
    const V = config.vocabSize
    expect(Array.from(a.data.subarray(0, 3 * V))).toEqual(Array.from(b.data.subarray(0, 3 * V)))
    expect(Array.from(a.data.subarray(3 * V, 4 * V))).not.toEqual(Array.from(b.data.subarray(3 * V, 4 * V)))
  })

  it('整個模型的梯度與數值微分一致（每個參數抽查數個元素）', () => {
    const model = testModel()
    const x = [0, 3, 7, 1, 4, 10], y = [3, 7, 1, 4, 10, 0]
    const lossFn = (): number => noGrad(() => model.loss(x, y).item())
    for (const p of model.parameters()) p.grad.fill(0)
    model.loss(x, y).backward()

    const rng = new RNG(5)
    const h = 1e-5
    let checked = 0
    for (const p of model.parameters()) {
      for (let n = 0; n < 4; n++) {
        const i = rng.randint(p.size)
        const original = p.data[i]
        p.data[i] = original + h
        const plus = lossFn()
        p.data[i] = original - h
        const minus = lossFn()
        p.data[i] = original
        const numeric = (plus - minus) / (2 * h)
        expect(Math.abs(numeric - p.grad[i])).toBeLessThan(1e-6 * Math.max(1, Math.abs(numeric)))
        checked++
      }
    }
    expect(checked).toBeGreaterThan(100)
  })

  it('能把一小批資料背起來（overfit）：訓練流程整體是通的', () => {
    const model = new GPT(config, new RNG(1))
    const optimizer = new AdamW(model.parameters())
    const batch = [
      { x: [0, 1, 2, 3, 4, 5], y: [1, 2, 3, 4, 5, 0] },
      { x: [0, 6, 7, 8, 9, 10], y: [6, 7, 8, 9, 10, 0] },
    ]
    // 兩條序列的第一題都是「<end> 之後接什麼」，答案各半，所以 loss 的理論下限是 ln(2)/6
    const floor = Math.log(2) / 6
    const before = noGrad(() => batchLoss(model, batch).item())
    let after = before
    for (let step = 0; step < 300; step++) after = trainStep(model, optimizer, batch, 0.01).loss
    expect(before).toBeGreaterThan(2)
    expect(after).toBeLessThan(floor + 0.02)

    // 從 <end> 開始、以 1 起頭 greedy 生成，應該原樣背出第一條序列
    const out = generate(model, [0, 1], { maxNewTokens: 6, temperature: 0, stopToken: 0, rng: new RNG() })
    expect(out).toEqual([0, 1, 2, 3, 4, 5])
  })

  it('checkpoint 存檔再讀回，輸出幾乎相同（權重存成 Float32，有微小誤差）', () => {
    const model = testModel()
    const tokenizer = new CharTokenizer('abcdefghij'.split(''))
    const restored = loadCheckpoint(JSON.parse(JSON.stringify(saveCheckpoint(model, tokenizer))))
    const a = noGrad(() => model.forward([1, 2, 3]))
    const b = noGrad(() => restored.model.forward([1, 2, 3]))
    for (let i = 0; i < a.size; i++) expect(b.data[i]).toBeCloseTo(a.data[i], 4)
    expect(restored.tokenizer.decode([1, 2, 3])).toBe('abc')
  })
})

describe('抽樣', () => {
  const logits = [1, 3, 2, 0]

  it('temperature = 0 時永遠選最大的', () => {
    expect(Array.from(samplingDistribution(logits, { temperature: 0 }))).toEqual([0, 1, 0, 0])
  })

  it('溫度越低，分布越集中在最高分的 token', () => {
    const cold = samplingDistribution(logits, { temperature: 0.5 })
    const hot = samplingDistribution(logits, { temperature: 2 })
    expect(cold[1]).toBeGreaterThan(hot[1])
    expect(cold.reduce((s, p) => s + p, 0)).toBeCloseTo(1, 12)
  })

  it('topK 只保留前 k 名', () => {
    const probs = samplingDistribution(logits, { topK: 2 })
    expect(probs[0]).toBe(0)
    expect(probs[3]).toBe(0)
    expect(probs[1] + probs[2]).toBeCloseTo(1, 12)
  })

  it('同樣的種子生成同樣的結果', () => {
    const model = testModel()
    const run = (): number[] => generate(model, [0], { maxNewTokens: 7, rng: new RNG(9) })
    expect(run()).toEqual(run())
  })
})
