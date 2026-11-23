// 把教學網頁打包成單一 HTML 檔：
//   引擎（src/index.ts，用 esbuild 打包成瀏覽器可用的全域變數 TinyGPT）
// ＋ 訓練好的模型（checkpoints/model.json）
// ＋ 小資料集（data/poems-small.txt，第六關在瀏覽器裡即時訓練用）
// ＋ 頁面與互動邏輯（web/src/）
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { build } from 'esbuild'
import { parseDocs } from '../src/data.ts'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const read = (path: string): string => readFileSync(join(root, path), 'utf8')

const bundle = await build({
  entryPoints: [join(root, 'src/index.ts')],
  bundle: true,
  format: 'iife',
  globalName: 'TinyGPT',
  target: 'es2022',
  minify: true,
  write: false,
})
const engine = bundle.outputFiles[0].text

const config = {
  /** 第四關試煉要找的注意力模式：往前數幾個位置（1 = 前一個字、6 = 上一句同位置） */
  attentionOffset: Number(process.env.ATTENTION_OFFSET ?? 1),
  /** 第六關可選的學習率與過關門檻 */
  trainLRs: [0.0001, 0.0005, 0.002, 0.005, 0.02, 0.1],
  trainTarget: 4.9,
}

// 內嵌在 <script> 裡的內容不能出現 "</script"，把 "<" 換成等價的跳脫字元
const json = (value: unknown): string => JSON.stringify(value).replaceAll('<', '\\u003c')
const data = (id: string, value: unknown): string => `<script type="application/json" id="${id}">${json(value)}</script>`
for (const [name, code] of [['engine', engine], ['app', read('web/src/app.js')]]) {
  if (code.includes('</script')) throw new Error(`${name} 裡含有 </script，無法內嵌`)
}

const fragment = [
  read('web/src/page.html'),
  data('data-ckpt', JSON.parse(read(process.env.CKPT ?? 'checkpoints/model.json'))),
  data('data-poems', parseDocs(read('data/poems-small.txt'))),
  data('data-config', config),
  `<script>${engine}</script>`,
  `<script>${read('web/src/app.js')}</script>`,
].join('\n')

// artifact.html：不含 <html> 外殼的片段（發布用）；index.html：可直接用瀏覽器開啟的完整檔案
const page = `<!doctype html>
<html lang="zh-Hant-TW">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>
</head>
<body>
${fragment}
</body>
</html>
`
mkdirSync(join(root, 'web/dist'), { recursive: true })
writeFileSync(join(root, 'web/dist/artifact.html'), fragment)
writeFileSync(join(root, 'web/dist/index.html'), page)
console.log(`web/dist/index.html：${(page.length / 1024 / 1024).toFixed(2)} MB（引擎 ${(engine.length / 1024).toFixed(1)} KB）`)
