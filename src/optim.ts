// 第四站：最佳化（optimizer）——拿到梯度之後，怎麼更新參數。
import type { Tensor } from './tensor.ts'

export interface AdamWOptions {
  beta1?: number
  beta2?: number
  eps?: number
  weightDecay?: number
}

/**
 * AdamW。
 *
 * 最單純的梯度下降是 w ← w − lr·g，但它有兩個問題：
 *   1. 每個 batch 的梯度都有雜訊，方向會抖來抖去。
 *   2. 不同參數的梯度大小可能差好幾個數量級，同一個 lr 很難同時適合所有參數。
 *
 * Adam 對每個參數維護兩個移動平均：
 *   m：梯度的平均（動量，momentum）→ 把雜訊平均掉，方向更穩。
 *   v：梯度平方的平均 → 估計「這個參數的梯度通常多大」。
 * 更新量是 m / √v：梯度一向很大的參數走小步，梯度一向很小的參數走大步。
 *
 * 「W」是 weight decay：每一步都把權重往 0 拉一點點，避免權重無限制長大（一種正則化）。
 * AdamW 的重點是把 decay 直接作用在權重上，而不是混進梯度裡。
 */
export class AdamW {
  private readonly params: Tensor[]
  private readonly m: Float64Array[]
  private readonly v: Float64Array[]
  private readonly beta1: number
  private readonly beta2: number
  private readonly eps: number
  private readonly weightDecay: number
  private step_ = 0

  constructor(params: Tensor[], options: AdamWOptions = {}) {
    this.params = params
    this.m = params.map((p) => new Float64Array(p.size))
    this.v = params.map((p) => new Float64Array(p.size))
    this.beta1 = options.beta1 ?? 0.9
    this.beta2 = options.beta2 ?? 0.99
    this.eps = options.eps ?? 1e-8
    this.weightDecay = options.weightDecay ?? 0.01
  }

  step(lr: number): void {
    this.step_++
    const { beta1, beta2, eps } = this
    // m 和 v 從 0 開始，前幾步會被低估；除以 (1 − βᵗ) 把它校正回來（bias correction）
    const correction1 = 1 - beta1 ** this.step_
    const correction2 = 1 - beta2 ** this.step_

    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i]
      const m = this.m[i], v = this.v[i]
      // 只對「矩陣」型的權重做 decay；bias 和 LayerNorm 的 gain（都是 [1, n]）不做
      const decay = p.rows > 1 ? this.weightDecay : 0
      for (let j = 0; j < p.size; j++) {
        const g = p.grad[j]
        m[j] = beta1 * m[j] + (1 - beta1) * g
        v[j] = beta2 * v[j] + (1 - beta2) * g * g
        const mHat = m[j] / correction1
        const vHat = v[j] / correction2
        p.data[j] -= lr * (mHat / (Math.sqrt(vHat) + eps) + decay * p.data[j])
      }
    }
  }

  /** 梯度是「累加」的（+=），所以每一步開始前要歸零 */
  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0)
  }
}

/**
 * Gradient clipping：如果所有梯度合起來的長度（L2 norm）超過 maxNorm，就等比例縮小。
 * 偶爾會遇到梯度特別大的 batch，不擋一下的話一步就可能把模型踢壞。
 * 回傳縮放前的 norm，方便觀察訓練狀況。
 */
export function clipGradNorm(params: Tensor[], maxNorm: number): number {
  let total = 0
  for (const p of params) for (let j = 0; j < p.size; j++) total += p.grad[j] * p.grad[j]
  const norm = Math.sqrt(total)
  if (norm > maxNorm) {
    const factor = maxNorm / norm
    for (const p of params) for (let j = 0; j < p.size; j++) p.grad[j] *= factor
  }
  return norm
}

export interface LRSchedule {
  /** 最高學習率 */
  baseLR: number
  /** 暖身步數 */
  warmupSteps: number
  totalSteps: number
  /** 最後降到 baseLR 的幾分之幾 */
  minRatio?: number
}

/**
 * 學習率排程：先線性暖身（warmup），再沿 cosine 曲線慢慢降下來。
 *   暖身：一開始參數是亂數、Adam 的統計量也還不準，步伐太大容易走歪。
 *   衰減：越接近終點越需要小步微調。
 */
export function learningRate(step: number, schedule: LRSchedule): number {
  const { baseLR, warmupSteps, totalSteps, minRatio = 0.1 } = schedule
  if (step < warmupSteps) return (baseLR * (step + 1)) / warmupSteps
  const progress = Math.min(1, (step - warmupSteps) / Math.max(1, totalSteps - warmupSteps))
  const cosine = 0.5 * (1 + Math.cos(Math.PI * progress)) // 從 1 平滑降到 0
  return baseLR * (minRatio + (1 - minRatio) * cosine)
}
