import { describe, expect, it } from 'vitest'
import {
  add, concatCols, crossEntropy, embedding, gelu, layerNorm, maskFuture, matmul, meanScalars, noGrad, scale,
  sliceCols, softmax, Tensor, transpose,
} from '../src/tensor.ts'
import { Value } from '../src/value.ts'
import { expectGradientsMatch, randomTensor, toScalar } from './helpers.ts'

describe('Tensor 運算的 forward 結果', () => {
  it('matmul', () => {
    const a = Tensor.from([[1, 2], [3, 4]])
    const b = Tensor.from([[5, 6, 7], [8, 9, 10]])
    expect(Array.from(matmul(a, b).data)).toEqual([21, 24, 27, 47, 54, 61])
  })

  it('add 會把列向量 broadcast 到每一列', () => {
    const out = add(Tensor.from([[1, 2], [3, 4]]), Tensor.from([[10, 20]]))
    expect(Array.from(out.data)).toEqual([11, 22, 13, 24])
  })

  it('softmax 每一列加總為 1，且不會因為數值很大而溢位', () => {
    const out = softmax(Tensor.from([[1000, 1001, 1002], [-5, 0, 5]]))
    expect(out.data[0] + out.data[1] + out.data[2]).toBeCloseTo(1, 12)
    expect(out.data[3] + out.data[4] + out.data[5]).toBeCloseTo(1, 12)
    expect(out.data[2]).toBeGreaterThan(out.data[1])
  })

  it('maskFuture + softmax：未來位置的權重必為 0', () => {
    const out = softmax(maskFuture(randomTensor(4, 4, 1)))
    for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) expect(out.data[i * 4 + j]).toBe(0)
    expect(out.data[0]).toBe(1) // 第一個 token 只能看自己
  })

  it('layerNorm 之後每一列平均為 0、變異數為 1', () => {
    const out = layerNorm(randomTensor(3, 8, 2), Tensor.filled(1, 8, 1), new Tensor(1, 8))
    for (let i = 0; i < 3; i++) {
      const row = Array.from(out.data.subarray(i * 8, i * 8 + 8))
      const mean = row.reduce((s, v) => s + v, 0) / 8
      const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / 8
      expect(mean).toBeCloseTo(0, 10)
      expect(variance).toBeCloseTo(1, 3)
    }
  })

  it('crossEntropy：均勻分布的 loss 是 ln(V)，答對且很有信心時接近 0', () => {
    expect(crossEntropy(new Tensor(2, 5), [1, 3]).item()).toBeCloseTo(Math.log(5), 12)
    expect(crossEntropy(Tensor.from([[0, 50, 0]]), [1]).item()).toBeCloseTo(0, 10)
  })

  it('noGrad 期間不會建立計算圖', () => {
    const a = randomTensor(2, 2, 3)
    const out = noGrad(() => matmul(a, a))
    expect(out.parents).toEqual([])
    expect(out.backwardFn).toBeNull()
  })
})

describe('Tensor 運算的梯度（與數值微分比對）', () => {
  const a = randomTensor(3, 4, 11)
  const b = randomTensor(4, 5, 12)
  const c = randomTensor(3, 4, 13)
  const row = randomTensor(1, 4, 14)
  const square = randomTensor(4, 4, 15)

  it('matmul', () => expectGradientsMatch([a, b], () => toScalar(matmul(a, b))))
  it('transpose', () => expectGradientsMatch([a], () => toScalar(transpose(a))))
  it('add（同形狀）', () => expectGradientsMatch([a, c], () => toScalar(add(a, c))))
  it('add（broadcast）', () => expectGradientsMatch([a, row], () => toScalar(add(a, row))))
  it('scale', () => expectGradientsMatch([a], () => toScalar(scale(a, -0.7))))
  it('sliceCols', () => expectGradientsMatch([a], () => toScalar(sliceCols(a, 1, 2))))
  it('concatCols', () => expectGradientsMatch([a, c], () => toScalar(concatCols([a, c, a]))))
  it('embedding（同一列被查多次）', () => expectGradientsMatch([b], () => toScalar(embedding(b, [2, 0, 2, 3]))))
  it('gelu', () => expectGradientsMatch([a], () => toScalar(gelu(a))))
  it('softmax', () => expectGradientsMatch([a], () => toScalar(softmax(a))))
  it('maskFuture + softmax', () => expectGradientsMatch([square], () => toScalar(softmax(maskFuture(square)))))
  it('crossEntropy', () => expectGradientsMatch([b], () => crossEntropy(b, [4, 0, 2, 2])))
  it('meanScalars', () =>
    expectGradientsMatch([a, c], () => meanScalars([toScalar(a, 1), toScalar(c, 2), toScalar(a, 3)])))

  it('layerNorm（輸入、gain、bias）', () => {
    const gain = randomTensor(1, 4, 16)
    const bias = randomTensor(1, 4, 17)
    expectGradientsMatch([a, gain, bias], () => toScalar(layerNorm(a, gain, bias)))
  })

  it('一個 tensor 在計算圖裡被用到多次時，梯度會累加', () => {
    expectGradientsMatch([square], () => toScalar(matmul(add(square, transpose(square)), square)))
  })
})

describe('Tensor 引擎與 Value 引擎互相驗證', () => {
  it('同一個算式，兩套引擎算出相同的值與梯度', () => {
    // y = Σ gelu 換成 tanh 不存在於 Tensor，所以比對 (A·B)·s 加總這種純線性代數的算式
    const A = [[0.5, -1.2], [2.0, 0.3]]
    const B = [[1.5, 0.7], [-0.4, 0.9]]
    const tA = Tensor.from(A), tB = Tensor.from(B)
    const ones = Tensor.filled(1, 2, 1), onesCol = Tensor.filled(2, 1, 1)
    const tOut = matmul(matmul(ones, scale(matmul(tA, tB), 3)), onesCol)
    tOut.backward()

    const vA = A.map((r) => r.map((x) => new Value(x)))
    const vB = B.map((r) => r.map((x) => new Value(x)))
    let vOut = new Value(0)
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        vOut = vOut.add(vA[i][0].mul(vB[0][j]).add(vA[i][1].mul(vB[1][j])).mul(3))
      }
    }
    vOut.backward()

    expect(tOut.item()).toBeCloseTo(vOut.data, 12)
    expect(Array.from(tA.grad)).toEqual(vA.flat().map((v) => expect.closeTo(v.grad, 12)))
    expect(Array.from(tB.grad)).toEqual(vB.flat().map((v) => expect.closeTo(v.grad, 12)))
  })
})
