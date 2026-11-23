import { describe, expect, it } from 'vitest'
import { Value } from '../src/value.ts'

describe('Value（純量 autograd）', () => {
  it('乘法與加法的梯度符合連鎖律', () => {
    // L = (a·b + c)²，手算：dL/da = 2(ab+c)·b、dL/db = 2(ab+c)·a、dL/dc = 2(ab+c)
    const a = new Value(2), b = new Value(-3), c = new Value(10)
    const L = a.mul(b).add(c).pow(2)
    L.backward()
    expect(L.data).toBe(16)
    expect(a.grad).toBe(2 * 4 * -3)
    expect(b.grad).toBe(2 * 4 * 2)
    expect(c.grad).toBe(2 * 4)
  })

  it('同一個值被用到多次時，梯度會累加', () => {
    const a = new Value(3)
    const y = a.mul(a).add(a) // y = a² + a → dy/da = 2a + 1
    y.backward()
    expect(a.grad).toBe(7)
  })

  it('各種運算的梯度都和數值微分一致', () => {
    const f = (x: Value): Value => x.tanh().mul(x.exp()).add(x.relu().div(x.pow(2).add(1))).sub(x.add(3).log())
    for (const x0 of [-1.5, 0.3, 2]) {
      const x = new Value(x0)
      f(x).backward()
      const h = 1e-6
      const numeric = (f(new Value(x0 + h)).data - f(new Value(x0 - h)).data) / (2 * h)
      expect(x.grad).toBeCloseTo(numeric, 6)
    }
  })

  it('用梯度下降可以訓練一個小型神經網路學會 XOR', () => {
    // 2 → 4 → 1 的 MLP，全部用 Value 組出來
    let seed = 1
    const rand = (): number => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1
    const w1 = Array.from({ length: 4 }, () => [new Value(rand()), new Value(rand())])
    const b1 = Array.from({ length: 4 }, () => new Value(0))
    const w2 = Array.from({ length: 4 }, () => new Value(rand()))
    const b2 = new Value(0)
    const params = [...w1.flat(), ...b1, ...w2, b2]
    const predict = (x: number[]): Value => {
      const hidden = w1.map((w, i) => w[0].mul(x[0]).add(w[1].mul(x[1])).add(b1[i]).tanh())
      return hidden.reduce((sum, h, i) => sum.add(h.mul(w2[i])), b2)
    }
    const data: [number[], number][] = [[[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]]
    let loss = 0
    for (let step = 0; step < 500; step++) {
      for (const p of params) p.grad = 0
      const L = data.reduce((sum, [x, y]) => sum.add(predict(x).sub(y).pow(2)), new Value(0))
      L.backward()
      for (const p of params) p.data -= 0.1 * p.grad
      loss = L.data
    }
    expect(loss).toBeLessThan(0.01)
  })
})
