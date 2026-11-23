// 存檔與讀檔：把模型設定、tokenizer 和所有權重打包成一個 JSON 物件。
// 權重以 Float32 存成 base64 字串，Node 和瀏覽器都能讀。
import { GPT } from './gpt.ts'
import type { GPTConfig } from './gpt.ts'
import { tokenizerFromJSON } from './tokenizer.ts'
import type { Tokenizer, TokenizerJSON } from './tokenizer.ts'

export interface Checkpoint {
  config: GPTConfig
  tokenizer: TokenizerJSON
  /** 所有參數依 model.parameters() 的順序串接，Float32 → base64 */
  weights: string
  /** 訓練過程的紀錄（loss 曲線、各階段的生成樣本等） */
  meta?: Record<string, unknown>
}

export function saveCheckpoint(model: GPT, tokenizer: Tokenizer, meta?: Record<string, unknown>): Checkpoint {
  const flat = new Float32Array(model.parameterCount())
  let offset = 0
  for (const p of model.parameters()) {
    flat.set(p.data, offset)
    offset += p.size
  }
  return { config: model.config, tokenizer: tokenizer.toJSON(), weights: toBase64(new Uint8Array(flat.buffer)), meta }
}

export function loadCheckpoint(checkpoint: Checkpoint): { model: GPT; tokenizer: Tokenizer } {
  const model = new GPT(checkpoint.config)
  const bytes = fromBase64(checkpoint.weights)
  const flat = new Float32Array(bytes.buffer, 0, bytes.byteLength / 4)
  if (flat.length !== model.parameterCount()) throw new Error('權重數量與模型設定不符')
  let offset = 0
  for (const p of model.parameters()) {
    p.data.set(flat.subarray(offset, offset + p.size))
    offset += p.size
  }
  return { model, tokenizer: tokenizerFromJSON(checkpoint.tokenizer) }
}

function toBase64(bytes: Uint8Array): string {
  let binary = ''
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  }
  return btoa(binary)
}

function fromBase64(text: string): Uint8Array {
  const binary = atob(text)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}
