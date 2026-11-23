// 第二站：矩陣版的自動微分引擎。
//
// 想法和 value.ts 完全一樣，只是每個節點從「一個數字」變成「一個矩陣」。
// 為了讓程式容易讀，這個專案裡所有 Tensor 都是 2D 矩陣（rows × cols）：
//   - 一段長度 T 的序列，每個 token 用 C 維向量表示 → [T, C]
//   - bias 這類向量 → [1, C]
//   - loss 這種純量 → [1, 1]
// 資料用一維的 Float64Array 以 row-major 順序存放：第 i 列第 j 行在 data[i * cols + j]。

let gradEnabled = true

/** 在 fn 執行期間不記錄計算圖（生成文字、算驗證 loss 時用，省時間也省記憶體） */
export function noGrad<T>(fn: () => T): T {
  const prev = gradEnabled
  gradEnabled = false
  try {
    return fn()
  } finally {
    gradEnabled = prev
  }
}

export class Tensor {
  readonly rows: number
  readonly cols: number
  readonly data: Float64Array
  /** d(loss) / d(data)，形狀和 data 相同 */
  readonly grad: Float64Array
  parents: Tensor[] = []
  backwardFn: (() => void) | null = null

  constructor(rows: number, cols: number, data?: ArrayLike<number>) {
    this.rows = rows
    this.cols = cols
    this.data = data ? Float64Array.from(data) : new Float64Array(rows * cols)
    if (this.data.length !== rows * cols) throw new Error(`資料長度 ${this.data.length} 與形狀 [${rows}, ${cols}] 不符`)
    this.grad = new Float64Array(rows * cols)
  }

  static from(matrix: number[][]): Tensor {
    return new Tensor(matrix.length, matrix[0].length, matrix.flat())
  }

  /** 以常態分布 N(0, std²) 初始化 */
  static randn(rows: number, cols: number, std: number, rng: { randn(): number }): Tensor {
    const t = new Tensor(rows, cols)
    for (let i = 0; i < t.data.length; i++) t.data[i] = rng.randn() * std
    return t
  }

  static filled(rows: number, cols: number, value: number): Tensor {
    const t = new Tensor(rows, cols)
    t.data.fill(value)
    return t
  }

  get size(): number {
    return this.data.length
  }

  /** 取出純量的值（只能用在 [1, 1]） */
  item(): number {
    if (this.size !== 1) throw new Error('item() 只能用在純量')
    return this.data[0]
  }

  /** 反向傳播：和 Value.backward() 一模一樣的拓撲排序 */
  backward(): void {
    const order: Tensor[] = []
    const visited = new Set<Tensor>()
    const visit = (t: Tensor): void => {
      if (visited.has(t)) return
      visited.add(t)
      for (const p of t.parents) visit(p)
      order.push(t)
    }
    visit(this)

    this.grad.fill(1)
    for (let i = order.length - 1; i >= 0; i--) order[i].backwardFn?.()
  }
}

/** 把運算結果掛進計算圖 */
function track(out: Tensor, parents: Tensor[], backwardFn: () => void): Tensor {
  if (gradEnabled) {
    out.parents = parents
    out.backwardFn = backwardFn
  }
  return out
}

// ───────────────────────────── 線性代數 ─────────────────────────────

/**
 * 矩陣乘法 out = a · b，形狀 [n, k] · [k, m] → [n, m]。
 * 反向：dA = dOut · Bᵀ、dB = Aᵀ · dOut。
 * 直覺：out[i][j] = Σₖ a[i][k]·b[k][j]，所以 a[i][k] 每變動一點，
 * 會透過 b[k][j] 影響到 out 的第 i 列整列。
 */
export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.cols !== b.rows) throw new Error(`matmul 形狀不符：[${a.rows}, ${a.cols}] · [${b.rows}, ${b.cols}]`)
  const n = a.rows, k = a.cols, m = b.cols
  const out = new Tensor(n, m)
  const A = a.data, B = b.data, O = out.data
  // 迴圈順序 i → p → j：最內層同時沿著 B 和 O 的「列」連續走，對 CPU 快取最友善
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const aip = A[i * k + p]
      const bRow = p * m, oRow = i * m
      for (let j = 0; j < m; j++) O[oRow + j] += aip * B[bRow + j]
    }
  }
  return track(out, [a, b], () => {
    const dO = out.grad, dA = a.grad, dB = b.grad
    for (let i = 0; i < n; i++) {
      for (let p = 0; p < k; p++) {
        const aip = A[i * k + p]
        const bRow = p * m, oRow = i * m
        let sum = 0
        for (let j = 0; j < m; j++) {
          const g = dO[oRow + j]
          sum += g * B[bRow + j]
          dB[bRow + j] += aip * g
        }
        dA[i * k + p] += sum
      }
    }
  })
}

/** 轉置。反向：把梯度轉置回去。 */
export function transpose(a: Tensor): Tensor {
  const out = new Tensor(a.cols, a.rows)
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) out.data[j * a.rows + i] = a.data[i * a.cols + j]
  }
  return track(out, [a], () => {
    for (let i = 0; i < a.rows; i++) {
      for (let j = 0; j < a.cols; j++) a.grad[i * a.cols + j] += out.grad[j * a.rows + i]
    }
  })
}

/**
 * 相加。b 可以和 a 同形狀，或是 [1, cols] 的列向量（會加到 a 的每一列，即 broadcast，用於 bias）。
 * 反向：加法把梯度原封不動往兩邊傳；被 broadcast 的 b 因為影響了每一列，要把每一列的梯度加總。
 */
export function add(a: Tensor, b: Tensor): Tensor {
  const broadcast = b.rows === 1 && a.rows > 1
  if (a.cols !== b.cols || (!broadcast && a.rows !== b.rows)) {
    throw new Error(`add 形狀不符：[${a.rows}, ${a.cols}] + [${b.rows}, ${b.cols}]`)
  }
  const out = new Tensor(a.rows, a.cols)
  const cols = a.cols
  for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] + b.data[broadcast ? i % cols : i]
  return track(out, [a, b], () => {
    for (let i = 0; i < out.size; i++) {
      a.grad[i] += out.grad[i]
      b.grad[broadcast ? i % cols : i] += out.grad[i]
    }
  })
}

/** 乘上常數 */
export function scale(a: Tensor, s: number): Tensor {
  const out = new Tensor(a.rows, a.cols)
  for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] * s
  return track(out, [a], () => {
    for (let i = 0; i < out.size; i++) a.grad[i] += out.grad[i] * s
  })
}

// ───────────────────────────── 切割與合併 ─────────────────────────────

/** 取出第 start 行起的 count 行（multi-head attention 用來把向量切成多個 head） */
export function sliceCols(a: Tensor, start: number, count: number): Tensor {
  const out = new Tensor(a.rows, count)
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < count; j++) out.data[i * count + j] = a.data[i * a.cols + start + j]
  }
  return track(out, [a], () => {
    for (let i = 0; i < a.rows; i++) {
      for (let j = 0; j < count; j++) a.grad[i * a.cols + start + j] += out.grad[i * count + j]
    }
  })
}

/** 把多個矩陣左右並排接起來（把各個 head 的輸出接回一個向量） */
export function concatCols(parts: Tensor[]): Tensor {
  const rows = parts[0].rows
  const cols = parts.reduce((sum, p) => sum + p.cols, 0)
  const out = new Tensor(rows, cols)
  let offset = 0
  for (const p of parts) {
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < p.cols; j++) out.data[i * cols + offset + j] = p.data[i * p.cols + j]
    }
    offset += p.cols
  }
  return track(out, parts, () => {
    let off = 0
    for (const p of parts) {
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < p.cols; j++) p.grad[i * p.cols + j] += out.grad[i * cols + off + j]
      }
      off += p.cols
    }
  })
}

/**
 * 查表：out 的第 t 列 = table 的第 ids[t] 列。
 * 反向：梯度只流回「被查到的那幾列」。同一列被查多次就累加多次。
 */
export function embedding(table: Tensor, ids: ArrayLike<number>): Tensor {
  const cols = table.cols
  const out = new Tensor(ids.length, cols)
  for (let t = 0; t < ids.length; t++) {
    const src = ids[t] * cols
    for (let j = 0; j < cols; j++) out.data[t * cols + j] = table.data[src + j]
  }
  return track(out, [table], () => {
    for (let t = 0; t < ids.length; t++) {
      const src = ids[t] * cols
      for (let j = 0; j < cols; j++) table.grad[src + j] += out.grad[t * cols + j]
    }
  })
}

// ───────────────────────────── 非線性與正規化 ─────────────────────────────

const GELU_C = Math.sqrt(2 / Math.PI)

/**
 * GELU 啟動函數（GPT-2 使用的 tanh 近似版）：gelu(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))。
 * 形狀像「平滑版的 ReLU」：負數大多被壓到接近 0，正數幾乎原樣通過。
 */
export function gelu(a: Tensor): Tensor {
  const out = new Tensor(a.rows, a.cols)
  for (let i = 0; i < out.size; i++) {
    const x = a.data[i]
    out.data[i] = 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)))
  }
  return track(out, [a], () => {
    for (let i = 0; i < out.size; i++) {
      const x = a.data[i]
      const t = Math.tanh(GELU_C * (x + 0.044715 * x * x * x))
      const dInner = GELU_C * (1 + 3 * 0.044715 * x * x)
      a.grad[i] += out.grad[i] * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dInner)
    }
  })
}

/**
 * Layer normalization：把每一列（每個 token 的向量）各自標準化成平均 0、變異數 1，
 * 再乘上可學習的 gain、加上可學習的 bias。
 *
 * 反向比較繞，因為一列裡的每個元素都會透過 mean 和 variance 影響同一列的所有輸出。
 * 令 x̂ = (x − μ)/σ、g = dOut·gain（對 x̂ 的梯度），推導結果是：
 *   dx = (g − mean(g) − x̂·mean(g·x̂)) / σ
 * 兩個被減掉的項，分別對應「經由 μ」與「經由 σ」的間接影響。
 */
export function layerNorm(x: Tensor, gain: Tensor, bias: Tensor, eps = 1e-5): Tensor {
  const rows = x.rows, cols = x.cols
  const out = new Tensor(rows, cols)
  const xhat = new Float64Array(rows * cols)
  const std = new Float64Array(rows)
  for (let i = 0; i < rows; i++) {
    const r = i * cols
    let mean = 0
    for (let j = 0; j < cols; j++) mean += x.data[r + j]
    mean /= cols
    let variance = 0
    for (let j = 0; j < cols; j++) variance += (x.data[r + j] - mean) ** 2
    variance /= cols
    std[i] = Math.sqrt(variance + eps)
    for (let j = 0; j < cols; j++) {
      xhat[r + j] = (x.data[r + j] - mean) / std[i]
      out.data[r + j] = xhat[r + j] * gain.data[j] + bias.data[j]
    }
  }
  return track(out, [x, gain, bias], () => {
    for (let i = 0; i < rows; i++) {
      const r = i * cols
      let meanG = 0, meanGX = 0
      for (let j = 0; j < cols; j++) {
        const dOut = out.grad[r + j]
        gain.grad[j] += dOut * xhat[r + j]
        bias.grad[j] += dOut
        const g = dOut * gain.data[j]
        meanG += g
        meanGX += g * xhat[r + j]
      }
      meanG /= cols
      meanGX /= cols
      for (let j = 0; j < cols; j++) {
        const g = out.grad[r + j] * gain.data[j]
        x.grad[r + j] += (g - meanG - xhat[r + j] * meanGX) / std[i]
      }
    }
  })
}

// ───────────────────────────── Attention 專用 ─────────────────────────────

/**
 * Causal mask：把「未來」的位置（行 j > 列 i）設成 −∞。
 * 之後做 softmax 時 e^(−∞) = 0，等於第 i 個 token 完全看不到它後面的 token。
 * 反向：被遮掉的位置對輸出沒有影響，梯度為 0；其餘原樣通過。
 */
export function maskFuture(a: Tensor): Tensor {
  const out = new Tensor(a.rows, a.cols)
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) out.data[i * a.cols + j] = j > i ? -Infinity : a.data[i * a.cols + j]
  }
  return track(out, [a], () => {
    for (let i = 0; i < a.rows; i++) {
      for (let j = 0; j <= i && j < a.cols; j++) a.grad[i * a.cols + j] += out.grad[i * a.cols + j]
    }
  })
}

/**
 * 對每一列做 softmax：pⱼ = e^(xⱼ) / Σₖ e^(xₖ)，把任意實數變成「加總為 1 的機率分布」。
 * 計算前先減掉該列最大值，避免 e^x 溢位（數學上結果不變）。
 * 反向：dxⱼ = pⱼ · (dOutⱼ − Σₖ dOutₖ·pₖ)。
 */
export function softmax(a: Tensor): Tensor {
  const rows = a.rows, cols = a.cols
  const out = new Tensor(rows, cols)
  for (let i = 0; i < rows; i++) softmaxRow(a.data, i * cols, cols, out.data)
  return track(out, [a], () => {
    for (let i = 0; i < rows; i++) {
      const r = i * cols
      let dot = 0
      for (let j = 0; j < cols; j++) dot += out.grad[r + j] * out.data[r + j]
      for (let j = 0; j < cols; j++) a.grad[r + j] += out.data[r + j] * (out.grad[r + j] - dot)
    }
  })
}

function softmaxRow(src: Float64Array, offset: number, cols: number, dst: Float64Array): void {
  let max = -Infinity
  for (let j = 0; j < cols; j++) if (src[offset + j] > max) max = src[offset + j]
  let sum = 0
  for (let j = 0; j < cols; j++) {
    const e = Math.exp(src[offset + j] - max)
    dst[offset + j] = e
    sum += e
  }
  for (let j = 0; j < cols; j++) dst[offset + j] /= sum
}

// ───────────────────────────── Loss ─────────────────────────────

/**
 * Cross-entropy loss：logits [T, V] 是每個位置對 V 個 token 的原始分數，
 * targets[t] 是第 t 個位置的正確答案。
 *   loss = 平均( −log softmax(logits[t])[targets[t]] )
 * 也就是「模型給正確答案的機率」取 −log 再平均：機率越接近 1，loss 越接近 0。
 *
 * 把 softmax 和 −log 合在一起算，反向會簡化成很漂亮的形式：
 *   d(loss)/d(logits) = (p − onehot(target)) / T
 * 「預測的機率分布」減「正確答案」，就是該修正的方向。
 */
export function crossEntropy(logits: Tensor, targets: ArrayLike<number>): Tensor {
  const rows = logits.rows, cols = logits.cols
  if (targets.length !== rows) throw new Error('targets 長度必須等於 logits 的列數')
  const probs = new Float64Array(rows * cols)
  let total = 0
  for (let t = 0; t < rows; t++) {
    softmaxRow(logits.data, t * cols, cols, probs)
    total -= Math.log(probs[t * cols + targets[t]] + 1e-300)
  }
  const out = new Tensor(1, 1, [total / rows])
  return track(out, [logits], () => {
    const g = out.grad[0] / rows
    for (let t = 0; t < rows; t++) {
      const r = t * cols
      for (let j = 0; j < cols; j++) logits.grad[r + j] += probs[r + j] * g
      logits.grad[r + targets[t]] -= g
    }
  })
}

/** 多個純量取平均（把一個 batch 裡每條序列的 loss 平均起來） */
export function meanScalars(items: Tensor[]): Tensor {
  let sum = 0
  for (const t of items) sum += t.item()
  const out = new Tensor(1, 1, [sum / items.length])
  return track(out, items, () => {
    for (const t of items) t.grad[0] += out.grad[0] / items.length
  })
}
