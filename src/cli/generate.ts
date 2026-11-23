// 生成指令：pnpm generate -- --prompt 春 --count 5 --temperature 0.8
import { readFileSync } from 'node:fs'
import { parseArgs } from 'node:util'
import { loadCheckpoint } from '../checkpoint.ts'
import type { Checkpoint } from '../checkpoint.ts'
import { generate } from '../gpt.ts'
import { RNG } from '../rng.ts'

const { values: args } = parseArgs({
  options: {
    ckpt: { type: 'string', default: 'checkpoints/model.json' },
    prompt: { type: 'string', default: '' },
    count: { type: 'string', default: '5' },
    temperature: { type: 'string', default: '0.8' },
    'top-k': { type: 'string', default: '40' },
    'max-tokens': { type: 'string' },
    seed: { type: 'string', default: String(Date.now() % 100000) },
  },
})

const { model, tokenizer } = loadCheckpoint(JSON.parse(readFileSync(args.ckpt, 'utf8')) as Checkpoint)
const rng = new RNG(Number(args.seed))
const start = [tokenizer.endToken, ...tokenizer.encode(args.prompt)]

for (let i = 0; i < Number(args.count); i++) {
  const ids = generate(model, start, {
    maxNewTokens: Number(args['max-tokens'] ?? model.config.blockSize - start.length + 1),
    temperature: Number(args.temperature),
    topK: Number(args['top-k']),
    stopToken: tokenizer.endToken,
    rng,
  })
  console.log(tokenizer.decode(ids))
}
