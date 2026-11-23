// 神經網路的基本積木。每個積木都只是「幾個可學習的 Tensor（參數）+ 一個 forward 函式」。
import type { RNG } from './rng.ts'
import { add, embedding, layerNorm, matmul, Tensor } from './tensor.ts'

/** 權重初始化的標準差（沿用 GPT-2 的設定） */
export const INIT_STD = 0.02

/** 線性層：y = x·W + b。把每個 token 的向量從 nIn 維「重新混合」成 nOut 維。 */
export class Linear {
  readonly weight: Tensor
  readonly bias: Tensor | null

  constructor(nIn: number, nOut: number, rng: RNG, options: { bias?: boolean; std?: number } = {}) {
    this.weight = Tensor.randn(nIn, nOut, options.std ?? INIT_STD, rng)
    this.bias = options.bias === false ? null : new Tensor(1, nOut)
  }

  forward(x: Tensor): Tensor {
    const y = matmul(x, this.weight)
    return this.bias ? add(y, this.bias) : y
  }

  parameters(): Tensor[] {
    return this.bias ? [this.weight, this.bias] : [this.weight]
  }
}

/** Embedding 表：每個 id 對應一列可學習的向量。forward 就是查表。 */
export class Embedding {
  readonly weight: Tensor

  constructor(count: number, dim: number, rng: RNG) {
    this.weight = Tensor.randn(count, dim, INIT_STD, rng)
  }

  forward(ids: ArrayLike<number>): Tensor {
    return embedding(this.weight, ids)
  }

  parameters(): Tensor[] {
    return [this.weight]
  }
}

/** Layer normalization：gain 初始為 1、bias 初始為 0，也就是一開始「只做標準化」。 */
export class LayerNorm {
  readonly gain: Tensor
  readonly bias: Tensor

  constructor(dim: number) {
    this.gain = Tensor.filled(1, dim, 1)
    this.bias = new Tensor(1, dim)
  }

  forward(x: Tensor): Tensor {
    return layerNorm(x, this.gain, this.bias)
  }

  parameters(): Tensor[] {
    return [this.gain, this.bias]
  }
}
