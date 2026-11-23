// 訓練指令：pnpm train -- --data data/poems.txt --steps 3000
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { parseArgs } from 'node:util'
import { saveCheckpoint } from '../checkpoint.ts'
import { Dataset, parseDocs, splitDocs } from '../data.ts'
import { generate, GPT } from '../gpt.ts'
import { AdamW, learningRate } from '../optim.ts'
import { RNG } from '../rng.ts'
import { BPETokenizer, CharTokenizer } from '../tokenizer.ts'
import type { Tokenizer } from '../tokenizer.ts'
import { estimateLoss, trainStep } from '../trainer.ts'

const { values: args } = parseArgs({
  options: {
    data: { type: 'string', default: 'data/poems.txt' },
    out: { type: 'string', default: 'checkpoints/model.json' },
    steps: { type: 'string', default: '3000' },
    'batch-size': { type: 'string', default: '32' },
    'block-size': { type: 'string' },
    'n-layer': { type: 'string', default: '2' },
    'n-head': { type: 'string', default: '4' },
    'n-embd': { type: 'string', default: '64' },
    lr: { type: 'string', default: '0.003' },
    'weight-decay': { type: 'string', default: '0.01' },
    seed: { type: 'string', default: '42' },
    tokenizer: { type: 'string', default: 'char' },
    'vocab-size': { type: 'string', default: '1024' },
    'eval-interval': { type: 'string', default: '250' },
  },
})

const rng = new RNG(Number(args.seed))
const docs = parseDocs(readFileSync(args.data, 'utf8'))
const { train, val } = splitDocs(docs, 0.1, rng)

const tokenizer: Tokenizer =
  args.tokenizer === 'bpe'
    ? BPETokenizer.train(train.slice(0, 2000), Number(args['vocab-size'])) // 只取一部分文件來學合併規則，夠用且快
    : CharTokenizer.fromTexts(docs)

const trainSet = new Dataset(train, tokenizer)
const valSet = new Dataset(val, tokenizer)
const blockSize = Number(args['block-size'] ?? Math.min(64, Math.max(trainSet.maxLength, valSet.maxLength)))

const model = new GPT(
  {
    vocabSize: tokenizer.vocabSize,
    blockSize,
    nLayer: Number(args['n-layer']),
    nHead: Number(args['n-head']),
    nEmbd: Number(args['n-embd']),
  },
  rng,
)
const optimizer = new AdamW(model.parameters(), { weightDecay: Number(args['weight-decay']) })

const steps = Number(args.steps)
const batchSize = Number(args['batch-size'])
const evalInterval = Number(args['eval-interval'])
const schedule = { baseLR: Number(args.lr), warmupSteps: Math.min(100, Math.floor(steps / 10)), totalSteps: steps }

console.log(`文件數：訓練 ${train.length}／驗證 ${val.length}　字表：${tokenizer.vocabSize}　context：${blockSize}`)
console.log(`模型參數：${model.parameterCount().toLocaleString()} 個`)
console.log(`隨機亂猜的 loss 應該接近 ln(${tokenizer.vocabSize}) = ${Math.log(tokenizer.vocabSize).toFixed(3)}\n`)

const sampleRng = new RNG(Number(args.seed) + 1)
const evalRng = new RNG(Number(args.seed) + 2)
const sample = (): string =>
  tokenizer.decode(
    generate(model, [tokenizer.endToken], {
      maxNewTokens: blockSize,
      temperature: 0.8,
      topK: 40,
      stopToken: tokenizer.endToken,
      rng: sampleRng,
    }),
  )

const history: { step: number; train: number; val: number; sample: string }[] = []
const lossCurve: number[] = []
let smoothed = 0
const startedAt = Date.now()

for (let step = 0; step < steps; step++) {
  const lr = learningRate(step, schedule)
  const { loss } = trainStep(model, optimizer, trainSet.batch(rng, batchSize, blockSize), lr)
  smoothed = step === 0 ? loss : 0.95 * smoothed + 0.05 * loss
  if (step % 10 === 0) lossCurve.push(Number(smoothed.toFixed(4)))

  if (step % evalInterval === 0 || step === steps - 1) {
    const valLoss = estimateLoss(model, valSet, evalRng)
    const text = sample()
    history.push({ step, train: Number(smoothed.toFixed(4)), val: Number(valLoss.toFixed(4)), sample: text })
    const msPerStep = (Date.now() - startedAt) / (step + 1)
    console.log(
      `step ${String(step).padStart(5)}　train ${smoothed.toFixed(3)}　val ${valLoss.toFixed(3)}　lr ${lr.toFixed(5)}　${msPerStep.toFixed(0)} ms/step　${text}`,
    )
  }
}

mkdirSync(dirname(args.out), { recursive: true })
writeFileSync(
  args.out,
  JSON.stringify(saveCheckpoint(model, tokenizer, { data: args.data, steps, batchSize, history, lossCurve })),
)
console.log(`\n已存檔：${args.out}（${((Date.now() - startedAt) / 1000).toFixed(0)} 秒）`)
