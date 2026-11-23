import { expect } from 'vitest'
import { RNG } from '../src/rng.ts'
import { matmul, Tensor } from '../src/tensor.ts'

/** 用固定的隨機權重把矩陣壓成純量，這樣每個輸出元素的梯度都會被檢查到 */
export function toScalar(t: Tensor, seed = 7): Tensor {
  const rng = new RNG(seed)
  const left = Tensor.randn(1, t.rows, 1, rng)
  const right = Tensor.randn(t.cols, 1, 1, rng)
  return matmul(matmul(left, t), right)
}

/**
 * Gradient check：autograd 算出的梯度，應該等於「把輸入挪動一點點、看輸出變多少」的數值微分。
 *   數值微分 ≈ (f(x + h) − f(x − h)) / 2h
 */
export function expectGradientsMatch(inputs: Tensor[], fn: () => Tensor, tolerance = 1e-6): void {
  for (const t of inputs) t.grad.fill(0)
  fn().backward()
  const h = 1e-5
  for (const t of inputs) {
    for (let i = 0; i < t.size; i++) {
      const original = t.data[i]
      t.data[i] = original + h
      const plus = fn().item()
      t.data[i] = original - h
      const minus = fn().item()
      t.data[i] = original
      const numeric = (plus - minus) / (2 * h)
      const scale = Math.max(1, Math.abs(numeric), Math.abs(t.grad[i]))
      expect(Math.abs(numeric - t.grad[i]) / scale, `元素 ${i}：數值 ${numeric}、autograd ${t.grad[i]}`).toBeLessThan(tolerance)
    }
  }
}

export function randomTensor(rows: number, cols: number, seed: number): Tensor {
  return Tensor.randn(rows, cols, 1, new RNG(seed))
}
