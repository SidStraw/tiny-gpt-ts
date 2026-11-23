// 資料準備：從 chinese-poetry 的《全唐詩》挑出五言詩，整理成「一行一首」的純文字檔。
//
// 篩選規則：
//   1. 只留五言絕句（2 聯）與五言律詩（4 聯）；律詩每 2 聯拆成一首「絕句形狀」的文件。
//   2. 只保留最常見的 N 個字；含有罕見字的詩直接丟掉。
//      字表越小，模型最後一層（要對每個字算分數）就越便宜，小模型也比較學得起來。
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const rawDir = join(root, 'data/raw')
const BASE = 'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97'

async function download(): Promise<void> {
  mkdirSync(rawDir, { recursive: true })
  for (let i = 0; i <= 57; i++) {
    const name = `poet.tang.${i * 1000}.json`
    const file = join(rawDir, name)
    if (existsSync(file)) continue
    process.stdout.write(`下載 ${name}\r`)
    const res = await fetch(`${BASE}/${name}`)
    if (!res.ok) throw new Error(`下載失敗：${name} (${res.status})`)
    writeFileSync(file, await res.text())
  }
}

const COUPLET = /^[一-鿿]{5}，[一-鿿]{5}。$/

function collect(): string[] {
  const docs = new Set<string>()
  for (const f of readdirSync(rawDir)) {
    if (!f.endsWith('.json')) continue
    const poems = JSON.parse(readFileSync(join(rawDir, f), 'utf8')) as { paragraphs?: string[] }[]
    for (const poem of poems) {
      const ps = poem.paragraphs ?? []
      if ((ps.length !== 2 && ps.length !== 4) || !ps.every((p) => COUPLET.test(p))) continue
      for (let i = 0; i < ps.length; i += 2) docs.add(ps[i] + ps[i + 1])
    }
  }
  return [...docs]
}

function keepCommon(docs: string[], vocabSize: number): string[] {
  const count = new Map<string, number>()
  for (const d of docs) for (const ch of d) count.set(ch, (count.get(ch) ?? 0) + 1)
  const common = new Set(
    [...count.entries()].sort((a, b) => b[1] - a[1]).slice(0, vocabSize).map(([ch]) => ch),
  )
  return docs.filter((d) => [...d].every((ch) => common.has(ch)))
}

await download()
const all = collect()
for (const [file, vocab] of [['poems.txt', 2000], ['poems-small.txt', 1000]] as const) {
  const kept = keepCommon(all, vocab)
  writeFileSync(join(root, 'data', file), kept.join('\n') + '\n')
  console.log(`${file}：${kept.length} 首（常用字表 ${vocab} 字，原始 ${all.length} 首）`)
}
