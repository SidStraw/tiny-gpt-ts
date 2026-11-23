// 訓練迴圈的核心：一個 step 做的四件事。
import type { Dataset, Example } from './data.ts'
import type { GPT } from './gpt.ts'
import { AdamW, clipGradNorm } from './optim.ts'
import type { RNG } from './rng.ts'
import { meanScalars, noGrad } from './tensor.ts'
import type { Tensor } from './tensor.ts'

/** 一個 batch 的 loss：每條序列各自跑一次 forward，再把 loss 平均 */
export function batchLoss(model: GPT, batch: Example[]): Tensor {
  return meanScalars(batch.map((example) => model.loss(example.x, example.y)))
}

/**
 * 一步訓練：
 *   1. forward：算出這個 batch 的 loss（同時建好計算圖）
 *   2. backward：算出 loss 對每個參數的梯度
 *   3. clip：梯度太大就縮小
 *   4. update：optimizer 依梯度更新參數
 */
export function trainStep(
  model: GPT,
  optimizer: AdamW,
  batch: Example[],
  lr: number,
  maxGradNorm = 1,
): { loss: number; gradNorm: number } {
  optimizer.zeroGrad()
  const loss = batchLoss(model, batch)
  loss.backward()
  const gradNorm = clipGradNorm(model.parameters(), maxGradNorm)
  optimizer.step(lr)
  return { loss: loss.item(), gradNorm }
}

/** 在資料集上估計平均 loss（不建計算圖、不更新參數） */
export function estimateLoss(model: GPT, dataset: Dataset, rng: RNG, batches = 4, batchSize = 32): number {
  return noGrad(() => {
    let total = 0
    for (let i = 0; i < batches; i++) {
      total += batchLoss(model, dataset.batch(rng, batchSize, model.config.blockSize)).item()
    }
    return total / batches
  })
}
