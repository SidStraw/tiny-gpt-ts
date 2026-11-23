// 教學網頁的互動邏輯。window.TinyGPT 是由 src/index.ts 打包而成的引擎，和 CLI 用的是同一份程式碼。
(() => {
'use strict'

const T = window.TinyGPT
const $ = (selector, root = document) => root.querySelector(selector)
const readJSON = (id) => JSON.parse(document.getElementById(id).textContent)

const CKPT = readJSON('data-ckpt')
const POEMS = readJSON('data-poems')
const CONFIG = readJSON('data-config')
const { model, tokenizer } = T.loadCheckpoint(CKPT)
const END = tokenizer.endToken

// ───────────────────────────── 小工具 ─────────────────────────────

function h(tag, props, ...kids) {
  const node = document.createElement(tag)
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue
    if (key === 'class') node.className = value
    else if (key === 'text') node.textContent = value
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value)
    else if (value === true) node.setAttribute(key, '')
    else if (value !== false && value != null) node.setAttribute(key, value)
  }
  for (const kid of kids.flat()) if (kid != null) node.append(kid)
  return node
}

const num = (n) => (n < 0 ? '−' : '') + Math.abs(n)
const pct = (p) => (p >= 0.0995 ? (p * 100).toFixed(0) : (p * 100).toFixed(1)) + '%'
const tokenLabel = (id) => (id === END ? '結束' : tokenizer.tokenText(id))

function feedback(node, kind, text) {
  node.className = 'feedback' + (kind ? ' ' + kind : '')
  node.textContent = text
}

function shake(node) {
  node.classList.remove('shake')
  void node.offsetWidth
  node.classList.add('shake')
}

function canEncode(text) {
  try {
    tokenizer.encode(text)
    return true
  } catch {
    return false
  }
}

/** 讀取目前佈景主題的顏色，回傳 [r, g, b] */
function themeColors() {
  const style = getComputedStyle(document.documentElement)
  const out = {}
  for (const name of ['paper', 'surface', 'sunken', 'ink', 'ink-soft', 'line', 'azurite', 'malachite', 'gamboge', 'cinnabar']) {
    const hex = style.getPropertyValue('--' + name).trim().replace('#', '')
    out[name] = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16))
  }
  return out
}
const rgb = (c, alpha = 1) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t))

/** 依顯示寬度與裝置像素比設定 canvas，回傳繪圖環境與邏輯尺寸；畫布不可見時回傳 null */
function setupCanvas(canvas) {
  if (!canvas.dataset.h) canvas.dataset.h = canvas.getAttribute('height')
  const height = Number(canvas.dataset.h)
  canvas.style.height = height + 'px'
  const width = canvas.clientWidth
  if (!width) return null
  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.round(width * dpr)
  canvas.height = Math.round(height * dpr)
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, width, height)
  return { ctx, width, height }
}

/** 取出 logits 的最後一列（對下一個 token 的評分） */
function lastRow(logits) {
  return logits.data.subarray((logits.rows - 1) * logits.cols)
}

function topK(probs, k) {
  return Array.from(probs, (p, id) => ({ id, p }))
    .filter((item) => item.p > 0)
    .sort((a, b) => b.p - a.p)
    .slice(0, k)
}

function barRow(label, p, scale, options = {}) {
  const tag = options.onclick ? 'button' : 'div'
  return h(
    tag,
    { class: 'bar' + (options.hit ? ' hit' : ''), onclick: options.onclick, type: options.onclick ? 'button' : null },
    h('span', { class: 'ch', text: label }),
    h('span', { class: 'track' }, h('span', { class: 'fill', style: `display:block;width:${Math.max(1, (p / scale) * 100).toFixed(1)}%` })),
    h('span', { class: 'pc', text: pct(p) }),
  )
}

// ───────────────────────────── 進度與導覽 ─────────────────────────────

const SEALS = ['字', '導', '降', '注', '疊', '煉', '詩']
const HASHES = ['guan-1', 'guan-2', 'guan-3', 'guan-4', 'guan-5', 'guan-6', 'guan-7', 'chushi']
const STORE_KEY = 'qiguan-gpt-seals'
const sections = HASHES.map((id) => document.getElementById(id))
const INIT = {}
const REDRAW = {}
const inited = {}
let current = 0

let earned = SEALS.map(() => false)
try {
  const saved = JSON.parse(localStorage.getItem(STORE_KEY) || '[]')
  earned = SEALS.map((_, i) => saved[i] === true)
} catch {}

function renderMap() {
  const map = $('#map')
  map.replaceChildren(
    ...SEALS.map((ch, i) =>
      h('button', {
        class: 'seal' + (earned[i] ? ' earned' : ''),
        text: ch,
        title: `第 ${i + 1} 關${earned[i] ? '（已過關）' : ''}`,
        'aria-label': `第 ${i + 1} 關${earned[i] ? '，已過關' : ''}`,
        'aria-current': current === i + 1 ? 'true' : null,
        onclick: () => go(i + 1),
      }),
    ),
    h('button', {
      class: 'seal' + (earned.every(Boolean) ? ' earned' : ''),
      text: '終',
      title: '結語',
      'aria-label': '結語',
      'aria-current': current === 8 ? 'true' : null,
      onclick: () => go(8),
    }),
  )
}

function pass(level) {
  const trial = $('#trial-' + level)
  if (earned[level - 1]) return
  earned[level - 1] = true
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(earned))
  } catch {}
  trial.classList.add('done', 'fresh')
  renderMap()
}

function show(level) {
  current = level
  sections.forEach((section, i) => (section.hidden = i !== level - 1))
  if (!inited[level]) {
    inited[level] = true
    if (level <= 7 && earned[level - 1]) $('#trial-' + level).classList.add('done')
    INIT[level]?.()
  }
  REDRAW[level]?.()
  renderMap()
}

function go(level) {
  const hash = '#' + HASHES[level - 1]
  if (location.hash !== hash) {
    try {
      location.hash = hash
    } catch {}
  }
  if (current !== level) {
    show(level)
    window.scrollTo(0, 0)
  }
}

window.addEventListener('hashchange', () => {
  const index = HASHES.indexOf(location.hash.slice(1))
  if (index >= 0 && current !== index + 1) {
    show(index + 1)
    window.scrollTo(0, 0)
  }
})
document.addEventListener('click', (event) => {
  const target = event.target.closest('[data-go]')
  if (target) go(Number(target.dataset.go))
})

let redrawTimer = 0
const redrawSoon = () => {
  clearTimeout(redrawTimer)
  redrawTimer = setTimeout(() => REDRAW[current]?.(), 60)
}
window.addEventListener('resize', redrawSoon)
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawSoon)
new MutationObserver(redrawSoon).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })

// ───────────────────────────── 第一關：Tokenizer ─────────────────────────────

INIT[1] = () => {
  const input = $('#l1-text'), out = $('#l1-tokens'), bytes = $('#l1-bytes'), note = $('#l1-note')
  const encoder = new TextEncoder()

  const render = () => {
    let unknown = 0, byteCount = 0
    const chips = Array.from(input.value, (ch) => {
      const known = canEncode(ch)
      if (!known) unknown++
      const raw = Array.from(encoder.encode(ch))
      byteCount += raw.length
      const sub = bytes.checked
        ? raw.map((b) => b.toString(16).padStart(2, '0')).join(' ')
        : known ? String(tokenizer.encode(ch)[0]) : '字表外'
      return h('span', { class: 'tok' + (!known && !bytes.checked ? ' bad' : '') }, h('b', { text: ch === ' ' ? '␣' : ch }), h('i', { text: sub }))
    })
    out.replaceChildren(...chips)
    note.textContent = bytes.checked
      ? `${chips.length} 個字 = ${byteCount} 個 byte。BPE 就是從這些 byte 開始合併。`
      : unknown ? `${unknown} 個字不在字表裡。` : '每個字下方是它的 token id。'
  }
  input.addEventListener('input', render)
  bytes.addEventListener('change', render)
  render()

  // BPE 試煉
  const TEXT = '吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮'
  const box = $('#l1-bpe'), fb = $('#l1-fb'), log = $('#l1-log'), count = $('#l1-count')
  let tokens, selected, correct, fresh

  const pairCounts = () => {
    const counts = new Map()
    for (let i = 0; i + 1 < tokens.length; i++) {
      const key = tokens[i] + '\u0001' + tokens[i + 1]
      counts.set(key, (counts.get(key) || 0) + 1)
    }
    return counts
  }

  const renderBPE = () => {
    box.replaceChildren(
      ...tokens.map((token, i) =>
        h('button', {
          type: 'button',
          class: 'tok' + (i === selected || i === selected + 1 ? ' sel' : '') + (fresh.has(token) ? ' merged' : ''),
          'aria-pressed': i === selected || i === selected + 1 ? 'true' : 'false',
          onclick: () => {
            selected = Math.min(i, tokens.length - 2)
            renderBPE()
          },
        }, h('b', { text: token })),
      ),
    )
    count.textContent = `目前 ${tokens.length} 個 token　已答對 ${correct} / 3`
  }

  const reset = () => {
    tokens = Array.from(TEXT)
    selected = -1
    correct = 0
    fresh = new Set()
    log.replaceChildren()
    feedback(fb, '', '')
    renderBPE()
  }

  $('#l1-merge').addEventListener('click', () => {
    const counts = pairCounts()
    const best = Math.max(...counts.values())
    if (best < 2) return feedback(fb, 'ok', '每一對都只出現一次，沒有東西值得合併了。BPE 在這裡停止。')
    if (selected < 0) return feedback(fb, 'no', '先點一個 token，選取它和右邊那個。')
    const a = tokens[selected], b = tokens[selected + 1]
    const mine = counts.get(a + '\u0001' + b)
    if (mine < best) {
      shake(box)
      return feedback(fb, 'no', `「${a}${b}」只出現 ${mine} 次，還有一對出現了 ${best} 次。再數數看。`)
    }
    const before = tokens.length
    const merged = []
    for (let i = 0; i < tokens.length; i++) {
      if (i + 1 < tokens.length && tokens[i] === a && tokens[i + 1] === b) {
        merged.push(a + b)
        i++
      } else merged.push(tokens[i])
    }
    tokens = merged
    fresh.add(a + b)
    selected = -1
    correct++
    log.prepend(h('div', {}, h('b', { text: `「${a}${b}」` }), ` 出現 ${mine} 次，合併成一個新 token。${before} → ${tokens.length} 個 token。`))
    feedback(fb, 'ok', correct >= 3 ? '過關。常一起出現的字被壓成了一個 token，這就是「子詞」的由來。' : '答對了。再來一次。')
    renderBPE()
    if (correct >= 3) pass(1)
  })
  $('#l1-reset').addEventListener('click', reset)
  reset()
}

// ───────────────────────────── 第二關：Autograd ─────────────────────────────

INIT[2] = () => {
  const svg = $('#l2-graph'), fb = $('#l2-fb'), question = $('#l2-q'), choices = $('#l2-choices')
  const sliders = { a: $('#l2-a'), b: $('#l2-b'), c: $('#l2-c') }
  const BOX = { w: 132, h: 62 }
  const NODES = {
    a: { x: 14, y: 12, label: 'a' },
    b: { x: 14, y: 94, label: 'b' },
    c: { x: 14, y: 176, label: 'c' },
    d: { x: 196, y: 53, label: 'd = a × b' },
    e: { x: 378, y: 115, label: 'e = d + c' },
    L: { x: 540, y: 115, label: 'L = e²' },
  }
  const EDGES = [['a', 'd'], ['b', 'd'], ['d', 'e'], ['c', 'e'], ['e', 'L']]
  const STEPS = [
    { node: 'e', up: 'L', local: (v) => 2 * v.e, rule: 'L = e²，局部導數是 2e' },
    { node: 'd', up: 'e', local: () => 1, rule: 'e = d + c，加法把梯度原封不動傳過來' },
    { node: 'c', up: 'e', local: () => 1, rule: 'e = d + c，加法把梯度原封不動傳過來' },
    { node: 'a', up: 'd', local: (v) => v.b, rule: 'd = a × b，乘法的局部導數是「對方的值」b' },
    { node: 'b', up: 'd', local: (v) => v.a, rule: 'd = a × b，乘法的局部導數是「對方的值」a' },
  ]
  let values, truth, revealed, stepIndex

  const compute = () => {
    // 用專案裡的 Value 引擎做 forward 與 backward，當作標準答案
    const a = new T.Value(Number(sliders.a.value)), b = new T.Value(Number(sliders.b.value)), c = new T.Value(Number(sliders.c.value))
    const d = a.mul(b), e = d.add(c), L = e.pow(2)
    L.backward()
    values = { a: a.data, b: b.data, c: c.data, d: d.data, e: e.data, L: L.data }
    truth = { a: a.grad, b: b.grad, c: c.grad, d: d.grad, e: e.grad, L: 1 }
  }

  const draw = () => {
    const currentNode = STEPS[stepIndex]?.node
    const edges = EDGES.map(([from, to]) => {
      const x1 = NODES[from].x + BOX.w, y1 = NODES[from].y + BOX.h / 2
      const x2 = NODES[to].x - 7, y2 = NODES[to].y + BOX.h / 2
      return `<line class="edge" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#l2-arrow)"></line>`
    }).join('')
    const nodes = Object.entries(NODES).map(([name, n]) => {
      const known = name in revealed
      const cls = 'node' + (name === currentNode ? ' cur' : known ? ' known' : '')
      return `<g class="${cls}">
        <rect x="${n.x}" y="${n.y}" width="${BOX.w}" height="${BOX.h}" rx="5"></rect>
        <text class="nm" x="${n.x + 10}" y="${n.y + 19}">${n.label}</text>
        <text x="${n.x + 10}" y="${n.y + 37}">值 ${num(values[name])}</text>
        <text class="gr${known ? '' : ' unknown'}" x="${n.x + 10}" y="${n.y + 54}">梯度 ${known ? num(revealed[name]) : '？'}</text>
      </g>`
    }).join('')
    svg.innerHTML = `<defs><marker id="l2-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="arrow" d="M0,0 L10,5 L0,10 z"></path></marker></defs>${edges}${nodes}`
  }

  const ask = () => {
    choices.replaceChildren()
    const step = STEPS[stepIndex]
    if (!step) {
      question.textContent = '全部算完了。'
      return
    }
    const upstream = revealed[step.up], local = step.local(values), answer = truth[step.node]
    question.textContent = `∂L/∂${step.node} = ？　（上游 ∂L/∂${step.up} = ${num(upstream)}）`
    // 干擾選項：只取局部導數、只取上游梯度、正負號弄反……
    const pool = [answer, local, upstream, -answer, upstream + local, answer + 1, answer * 2, 0, answer - 2]
    const options = [...new Set(pool)].slice(0, 3)
    for (let i = options.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[options[i], options[j]] = [options[j], options[i]]
    }
    for (const option of options) {
      choices.append(h('button', {
        type: 'button', class: 'btn', text: num(option),
        onclick: () => {
          if (option !== answer) {
            shake(choices)
            return feedback(fb, 'no', '再想想：這個節點的梯度 = 上游傳來的梯度 × 這一步的局部導數。')
          }
          revealed[step.node] = answer
          stepIndex++
          feedback(fb, 'ok', `${step.rule}：${num(upstream)} × ${num(local)} = ${num(answer)}`)
          draw()
          ask()
          if (stepIndex >= STEPS.length) {
            feedback(fb, 'ok', `${step.rule}：${num(upstream)} × ${num(local)} = ${num(answer)}。五個梯度都和 value.ts 引擎算出的一致，過關。`)
            pass(2)
          }
        },
      }))
    }
  }

  const restart = () => {
    // e = 0 時所有梯度都是 0，題目會失去意義，稍微挪一下 c
    if (Number(sliders.a.value) * Number(sliders.b.value) + Number(sliders.c.value) === 0) {
      sliders.c.value = String(Number(sliders.c.value) + 1)
    }
    for (const key of ['a', 'b', 'c']) $(`#l2-${key}-out`).textContent = num(Number(sliders[key].value))
    compute()
    revealed = { L: 1 }
    stepIndex = 0
    feedback(fb, '', '')
    draw()
    ask()
  }

  for (const slider of Object.values(sliders)) slider.addEventListener('input', restart)
  $('#l2-new').addEventListener('click', () => {
    const pick = (min, max) => String(min + Math.floor(Math.random() * (max - min + 1)))
    let a = '0', b = '0'
    while (a === '0') a = pick(-4, 4)
    while (b === '0') b = pick(-4, 4)
    sliders.a.value = a
    sliders.b.value = b
    sliders.c.value = pick(-6, 6)
    restart()
  })
  restart()
}

// ───────────────────────────── 第三關：梯度下降 ─────────────────────────────

INIT[3] = () => {
  const XS = [0, 1, 2, 3, 4], YS = [0.6, 1.9, 3.6, 4.9, 6.6]
  const START = { w: -0.5, b: 3 }, BEST = { w: 1.5, b: 0.52 }
  const RANGE = { w: [-1.5, 4], b: [-3.5, 4.5] }
  const land = $('#l3-land'), fit = $('#l3-fit'), stat = $('#l3-stat'), fb = $('#l3-fb')
  const lrSlider = $('#l3-lr'), lrOut = $('#l3-lr-out')
  let state, useAdam = false, timer = 0

  const lossAt = (w, b) => XS.reduce((sum, x, i) => sum + (w * x + b - YS[i]) ** 2, 0) / XS.length
  // 梯度交給 Value 引擎算：L = 平均((w·x + b − y)²)
  const gradAt = (w, b) => {
    const W = new T.Value(w), B = new T.Value(b)
    let L = new T.Value(0)
    XS.forEach((x, i) => {
      L = L.add(W.mul(x).add(B).sub(YS[i]).pow(2))
    })
    L = L.div(XS.length)
    L.backward()
    return { loss: L.data, gw: W.grad, gb: B.grad }
  }
  const lr = () => 0.005 * 200 ** (Number(lrSlider.value) / 100)

  const draw = () => {
    const colors = themeColors()
    const a = setupCanvas(land)
    if (a) {
      const { ctx, width, height } = a
      const nx = 96, ny = 72
      const toX = (w) => ((w - RANGE.w[0]) / (RANGE.w[1] - RANGE.w[0])) * width
      const toY = (b) => height - ((b - RANGE.b[0]) / (RANGE.b[1] - RANGE.b[0])) * height
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < ny; j++) {
          const w = RANGE.w[0] + ((i + 0.5) / nx) * (RANGE.w[1] - RANGE.w[0])
          const b = RANGE.b[0] + ((j + 0.5) / ny) * (RANGE.b[1] - RANGE.b[0])
          const t = Math.min(1, Math.max(0, (Math.log10(lossAt(w, b) + 0.01) + 2) / 4.3))
          const band = Math.floor(t * 13) / 13 // 分成色階，看起來像等高線
          ctx.fillStyle = rgb(mix(colors.azurite, colors.paper, 0.12 + band * 0.88))
          ctx.fillRect((i / nx) * width, height - ((j + 1) / ny) * height, width / nx + 1, height / ny + 1)
        }
      }
      ctx.font = '11px ' + getComputedStyle(document.body).fontFamily
      ctx.fillStyle = rgb(colors.ink)
      ctx.fillText('b ↑', 6, 14)
      ctx.fillText('w →', width - 30, height - 6)
      ctx.font = 'bold 15px sans-serif'
      ctx.fillStyle = rgb(colors.paper)
      ctx.fillText('✕', toX(BEST.w) - 6, toY(BEST.b) + 5)
      ctx.strokeStyle = rgb(colors.gamboge)
      ctx.lineWidth = 2
      ctx.beginPath()
      state.path.forEach((p, i) => (i ? ctx.lineTo(toX(p.w), toY(p.b)) : ctx.moveTo(toX(p.w), toY(p.b))))
      ctx.stroke()
      state.path.forEach((p, i) => {
        const last = i === state.path.length - 1
        ctx.beginPath()
        ctx.arc(toX(p.w), toY(p.b), last ? 5.5 : 2.5, 0, Math.PI * 2)
        ctx.fillStyle = rgb(last ? colors.cinnabar : colors.gamboge)
        ctx.fill()
        if (last) {
          ctx.lineWidth = 1.5
          ctx.strokeStyle = rgb(colors.paper)
          ctx.stroke()
        }
      })
    }
    const f = setupCanvas(fit)
    if (f) {
      const { ctx, width, height } = f
      const toX = (x) => 30 + ((x + 0.5) / 5) * (width - 44)
      const toY = (y) => height - 24 - ((y + 2) / 11) * (height - 38)
      ctx.strokeStyle = rgb(colors.line)
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(toX(-0.5), toY(0)); ctx.lineTo(toX(4.5), toY(0))
      ctx.moveTo(toX(0), toY(-2)); ctx.lineTo(toX(0), toY(9))
      ctx.stroke()
      ctx.strokeStyle = rgb(colors.gamboge)
      ctx.lineWidth = 2.5
      ctx.beginPath()
      ctx.moveTo(toX(-0.5), toY(state.w * -0.5 + state.b)); ctx.lineTo(toX(4.5), toY(state.w * 4.5 + state.b))
      ctx.stroke()
      XS.forEach((x, i) => {
        // 每個點到直線的垂直距離就是誤差
        ctx.strokeStyle = rgb(colors.cinnabar, 0.7)
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(toX(x), toY(YS[i])); ctx.lineTo(toX(x), toY(state.w * x + state.b))
        ctx.stroke()
        ctx.beginPath()
        ctx.arc(toX(x), toY(YS[i]), 4.5, 0, Math.PI * 2)
        ctx.fillStyle = rgb(colors.ink)
        ctx.fill()
      })
      ctx.font = '12px ' + getComputedStyle(document.body).fontFamily
      ctx.fillStyle = rgb(colors['ink-soft'])
      ctx.fillText(`y = ${state.w.toFixed(2)}·x ${state.b < 0 ? '−' : '+'} ${Math.abs(state.b).toFixed(2)}`, 38, 18)
    }
    const g = gradAt(state.w, state.b)
    stat.replaceChildren(
      h('span', {}, '步數 ', h('b', { text: String(state.steps) })),
      h('span', {}, 'loss ', h('b', { text: g.loss < 1e4 ? g.loss.toFixed(4) : g.loss.toExponential(1) })),
      h('span', {}, '∂L/∂w ', h('b', { text: g.gw.toFixed(2) })),
      h('span', {}, '∂L/∂b ', h('b', { text: g.gb.toFixed(2) })),
    )
    lrOut.textContent = lr().toFixed(3)
  }

  const reset = () => {
    clearInterval(timer)
    timer = 0
    $('#l3-auto').textContent = '自動走'
    const params = new T.Tensor(1, 2, [START.w, START.b])
    state = { ...START, steps: 0, path: [{ ...START }], params, adam: new T.AdamW([params], { weightDecay: 0 }), over: false }
    feedback(fb, '', '還沒出發。')
    draw()
  }

  const step = () => {
    if (state.over) return false
    const g = gradAt(state.w, state.b)
    if (useAdam) {
      state.params.grad[0] = g.gw
      state.params.grad[1] = g.gb
      state.adam.step(lr())
      state.w = state.params.data[0]
      state.b = state.params.data[1]
    } else {
      state.w -= lr() * g.gw
      state.b -= lr() * g.gb
      state.params.data[0] = state.w
      state.params.data[1] = state.b
    }
    state.steps++
    state.path.push({ w: state.w, b: state.b })
    const loss = lossAt(state.w, state.b)
    if (!Number.isFinite(loss) || loss > 1e5) {
      state.over = true
      feedback(fb, 'no', '發散了。步伐太大，每一步都跨過谷底、落在更高的地方。回到起點，把學習率調小。')
    } else if (loss < 0.1 && state.steps <= 30) {
      state.over = true
      feedback(fb, 'ok', `第 ${state.steps} 步，loss ${loss.toFixed(3)}，過關。學習率夠大才走得快，又不能大到發散。`)
      pass(3)
    } else if (state.steps >= 30 && loss >= 0.1) {
      feedback(fb, 'no', `已經 ${state.steps} 步，loss 還有 ${loss.toFixed(3)}。回到起點，換一個學習率。`)
      if (state.steps >= 60) state.over = true
    } else {
      feedback(fb, '', `第 ${state.steps} 步，loss ${loss.toFixed(3)}。`)
    }
    draw()
    return !state.over
  }

  $('#l3-step').addEventListener('click', step)
  $('#l3-auto').addEventListener('click', () => {
    if (timer) {
      clearInterval(timer)
      timer = 0
      $('#l3-auto').textContent = '自動走'
      return
    }
    $('#l3-auto').textContent = '暫停'
    timer = setInterval(() => {
      if (!step()) {
        clearInterval(timer)
        timer = 0
        $('#l3-auto').textContent = '自動走'
      }
    }, 110)
  })
  $('#l3-reset').addEventListener('click', reset)
  lrSlider.addEventListener('input', draw)
  const setOptimizer = (adam) => {
    useAdam = adam
    $('#l3-sgd').classList.toggle('on', !adam)
    $('#l3-adam').classList.toggle('on', adam)
    reset()
  }
  $('#l3-sgd').addEventListener('click', () => setOptimizer(false))
  $('#l3-adam').addEventListener('click', () => setOptimizer(true))
  REDRAW[3] = draw
  reset()
}

// ───────────────────────────── 第四關：Attention ─────────────────────────────

INIT[4] = () => {
  const FAMOUS = ['春眠不覺曉，處處聞啼鳥。夜來風雨聲，花落知多少。', '白日依山盡，黃河入海流。欲窮千里目，更上一層樓。', '床前明月光，疑是地上霜。舉頭望明月，低頭思故鄉。']
    .filter((poem) => canEncode(poem))
  const { nLayer, nHead } = model.config
  const toks = $('#l4-toks'), heads = $('#l4-heads'), top = $('#l4-top'), heat = $('#l4-heat'), title = $('#l4-title'), fb = $('#l4-fb')
  let poem = FAMOUS[0] || POEMS[0]
  let ids, attention, layer = 0, head = 0, query = 11

  /** 跑一次 forward，取出每一層、每個 head 的注意力矩陣 */
  const attend = (text) => {
    const tokenIds = [END, ...tokenizer.encode(text)].slice(0, model.config.blockSize)
    T.noGrad(() => model.forward(tokenIds))
    return { tokenIds, maps: model.blocks.map((block) => block.attn.lastAttention.map((t) => ({ n: t.rows, data: Float64Array.from(t.data) }))) }
  }

  // 試煉的答案由模型自己決定：統計每個 head 平均放多少注意力在「往前數 offset 個」的位置
  const OFFSET = CONFIG.attentionOffset
  const TARGET = OFFSET === 1 ? '前一個字' : '上一句同一個位置的字'
  const score = Array.from({ length: nLayer }, () => new Float64Array(nHead))
  {
    const sample = POEMS.slice(0, 24)
    let count = 0
    for (const text of sample) {
      const { maps } = attend(text)
      const n = maps[0][0].n
      for (let l = 0; l < nLayer; l++) {
        for (let hd = 0; hd < nHead; hd++) {
          for (let i = OFFSET; i < n; i++) score[l][hd] += maps[l][hd].data[i * n + i - OFFSET]
        }
      }
      count += n - OFFSET
    }
    for (const row of score) for (let hd = 0; hd < nHead; hd++) row[hd] /= count
  }
  const bestScore = Math.max(...score.map((row) => Math.max(...row)))
  $('#l4-goal').textContent = `兩層共 ${nLayer * nHead} 個 head，找出最專注於「${TARGET}」的那一個。`

  const render = () => {
    const map = attention[layer][head], n = map.n
    const row = map.data.subarray(query * n, query * n + n)
    const max = Math.max(...row)
    const chip = (j) =>
      h('button', {
        type: 'button',
        class: 'tok' + (j === query ? ' query' : '') + (j > query ? ' masked' : ''),
        style: j <= query ? `background: color-mix(in srgb, var(--azurite) ${((row[j] / max) * 55).toFixed(0)}%, var(--paper))` : null,
        title: j > query ? '被 causal mask 遮住' : `注意力 ${pct(row[j])}`,
        onclick: () => {
          query = j
          render()
        },
      }, h('b', { text: j === 0 ? '始' : tokenizer.tokenText(ids[j]) }), h('i', { text: j > query ? '遮' : pct(row[j]) }))
    const rows = [[0]]
    for (let start = 1; start < n; start += 6) rows.push(Array.from({ length: Math.min(6, n - start) }, (_, k) => start + k))
    toks.replaceChildren(...rows.map((indexes) => h('div', { class: 'toks' }, indexes.map(chip))))

    heads.replaceChildren(
      ...Array.from({ length: nLayer * nHead }, (_, i) => {
        const l = Math.floor(i / nHead), hd = i % nHead
        return h('button', {
          type: 'button', class: 'btn' + (l === layer && hd === head ? ' on' : ''), text: `第 ${l + 1} 層 · head ${hd + 1}`,
          onclick: () => {
            layer = l
            head = hd
            render()
          },
        })
      }),
    )

    const name = query === 0 ? '開頭符號' : `「${tokenizer.tokenText(ids[query])}」`
    title.textContent = `${name}（第 ${query} 個位置）最注意的字`
    const ranked = Array.from(row, (p, j) => ({ j, p })).sort((a, b) => b.p - a.p).slice(0, 5)
    top.replaceChildren(...ranked.map(({ j, p }) => barRow(j === 0 ? '始' : tokenizer.tokenText(ids[j]), p, 1, { hit: j === query })))
    drawHeat()
  }

  const drawHeat = () => {
    const setup = setupCanvas(heat)
    if (!setup) return
    const { ctx, width, height } = setup
    const colors = themeColors()
    const map = attention[layer][head], n = map.n
    const size = Math.min(width, height) - 8, cell = size / n, x0 = (width - size) / 2, y0 = 4
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        ctx.fillStyle = j > i ? rgb(colors.sunken) : rgb(mix(colors.paper, colors.azurite, Math.min(1, map.data[i * n + j] * 1.6)))
        ctx.fillRect(x0 + j * cell, y0 + i * cell, cell - 0.5, cell - 0.5)
      }
    }
    ctx.strokeStyle = rgb(colors.gamboge)
    ctx.lineWidth = 2
    ctx.strokeRect(x0 - 1, y0 + query * cell - 1, size + 1, cell + 1)
  }

  const load = (text) => {
    poem = text
    const result = attend(text)
    ids = result.tokenIds
    attention = result.maps
    query = Math.min(query, ids.length - 1)
    render()
  }

  $('#l4-poems').replaceChildren(
    ...FAMOUS.map((text) => h('button', { type: 'button', class: 'btn', text: text.slice(0, 5), onclick: () => load(text) })),
    h('button', { type: 'button', class: 'btn', text: '隨機換一首', onclick: () => load(POEMS[Math.floor(Math.random() * POEMS.length)]) }),
  )

  $('#l4-answer').addEventListener('click', () => {
    const mine = score[layer][head]
    if (mine >= bestScore * 0.9) {
      feedback(fb, 'ok', `過關。這個 head 平均把 ${pct(mine)} 的注意力放在${TARGET}上。沒有人教它這麼做，是訓練時自己發現這樣最有助於預測。`)
      pass(4)
    } else {
      shake($('#l4-answer'))
      feedback(fb, 'no', `第 ${layer + 1} 層 head ${head + 1} 平均只把 ${pct(mine)} 的注意力放在${TARGET}上，有一個 head 高達 ${pct(bestScore)}。`)
    }
  })

  REDRAW[4] = drawHeat
  load(poem)
}

// ───────────────────────────── 第五關：組裝 ─────────────────────────────

INIT[5] = () => {
  const PIECES = [
    { kind: 'emb', label: 'Token embedding ＋ Position embedding', shape: 'ids → [T, C]' },
    { kind: 'ln', label: 'LayerNorm', shape: '[T, C]' },
    { kind: 'attn', label: 'Attention', shape: '[T, C]' },
    { kind: 'add', label: '加回主幹（殘差連接）', shape: '[T, C]' },
    { kind: 'ln', label: 'LayerNorm', shape: '[T, C]' },
    { kind: 'mlp', label: 'MLP', shape: '[T, C]' },
    { kind: 'add', label: '加回主幹（殘差連接）', shape: '[T, C]' },
    { kind: 'ln', label: 'LayerNorm', shape: '[T, C]' },
    { kind: 'out', label: '乘上 embedding 表的轉置', shape: '[T, C] → [T, V]' },
  ]
  const HINTS = {
    emb: '模型拿到的是整數 id，還不是向量。第一步要先查表。',
    ln: '進子層或輸出之前，要先把向量標準化。',
    attn: '標準化之後，先讓 token 之間交換資訊。',
    add: '子層算完了。它的結果要加回主幹，不能直接取代。',
    mlp: '資訊搬過來了，接著讓每個 token 各自消化。',
    out: 'Block 都跑完、也標準化了。最後把向量變成對每個字的評分。',
  }
  const pipe = $('#l5-pipe'), tray = $('#l5-tray'), fb = $('#l5-fb')
  let placed, order

  const render = () => {
    pipe.replaceChildren(
      ...(placed
        ? PIECES.slice(0, placed).map((p) => h('div', { class: 'piece' }, h('span', { text: p.label }), h('small', { text: p.shape })))
        : [h('div', { class: 'empty', text: '還是空的。資料進來的第一站是什麼？' })]),
    )
    tray.replaceChildren(
      ...order.filter((index) => !index.used).map((item) =>
        h('button', {
          type: 'button', class: 'btn', text: PIECES[item.i].label,
          onclick: (event) => {
            const expected = PIECES[placed].kind
            if (PIECES[item.i].kind !== expected) {
              shake(event.currentTarget)
              return feedback(fb, 'no', HINTS[expected])
            }
            item.used = true
            placed++
            feedback(fb, 'ok', placed === PIECES.length ? '過關。第 2 到 7 步就是一個 Block；真正的模型只是把它重複很多層。' : '對。')
            render()
            if (placed === PIECES.length) pass(5)
          },
        }),
      ),
    )
  }
  const reset = () => {
    placed = 0
    order = PIECES.map((_, i) => ({ i, used: false }))
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[order[i], order[j]] = [order[j], order[i]]
    }
    feedback(fb, '', '')
    render()
  }
  $('#l5-reset').addEventListener('click', reset)
  reset()

  // 逐層偷看預測
  const input = $('#l5-text'), lens = $('#l5-lens'), note = $('#l5-note')
  const KNOWN = ['春眠不覺曉，處處聞啼鳥。夜來風雨聲，花落知多少。', '白日依山盡，黃河入海流。欲窮千里目，更上一層樓。', '床前明月光，疑是地上霜。舉頭望明月，低頭思故鄉。']
  const renderLens = () => {
    const text = input.value
    if (!canEncode(text)) {
      lens.replaceChildren()
      note.textContent = '裡面有字表外的字，換一個試試。'
      return
    }
    const ids = [END, ...tokenizer.encode(text)].slice(-model.config.blockSize)
    const source = text && KNOWN.find((poem) => poem.startsWith(text) && poem.length > text.length)
    const answer = source ? tokenizer.encode(source[text.length])[0] : -1
    const stages = T.noGrad(() => {
      const positions = ids.map((_, i) => i)
      let x = T.add(model.wte.forward(ids), model.wpe.forward(positions))
      const residuals = [x]
      for (const block of model.blocks) residuals.push((x = block.forward(x)))
      // 把每一層之後的主幹直接拿去做輸出層的運算
      return residuals.map((r) => T.samplingDistribution(lastRow(T.matmul(model.lnF.forward(r), T.transpose(model.wte.weight)))))
    })
    lens.replaceChildren(
      ...stages.map((probs, i) => {
        const ranked = topK(probs, 5)
        return h('div', {},
          h('h3', { text: i === 0 ? '剛查完 embedding' : `第 ${i} 層之後` + (i === stages.length - 1 ? '（最終輸出）' : '') }),
          h('div', { class: 'bars' }, ranked.map(({ id, p }) => barRow(tokenLabel(id), p, ranked[0].p, { hit: id === answer }))),
        )
      }),
    )
    note.textContent = source ? `原詩的下一個字是「${source[text.length]}」（黃色）。模型沒背過這首詩，猜不中很正常；重點是看分布怎麼變。` : '長條的長度相對於該欄的第一名。'
  }
  input.addEventListener('input', renderLens)
  renderLens()
}

// ───────────────────────────── 損失曲線 ─────────────────────────────

function drawChart(canvas, cfg) {
  const setup = setupCanvas(canvas)
  if (!setup) return
  const { ctx, width, height } = setup
  const colors = themeColors()
  const pad = { l: 40, r: 12, t: 14, b: 26 }
  const toX = (x) => pad.l + (x / cfg.maxX) * (width - pad.l - pad.r)
  const toY = (y) => pad.t + ((cfg.maxY - Math.min(cfg.maxY, Math.max(cfg.minY, y))) / (cfg.maxY - cfg.minY)) * (height - pad.t - pad.b)
  const font = getComputedStyle(document.body).fontFamily
  ctx.font = '11px ' + font
  ctx.lineWidth = 1
  for (let y = Math.ceil(cfg.minY); y <= cfg.maxY; y++) {
    ctx.strokeStyle = rgb(colors.line)
    ctx.beginPath()
    ctx.moveTo(pad.l, toY(y)); ctx.lineTo(width - pad.r, toY(y))
    ctx.stroke()
    ctx.fillStyle = rgb(colors['ink-soft'])
    ctx.textAlign = 'right'
    ctx.fillText(String(y), pad.l - 6, toY(y) + 4)
  }
  ctx.textAlign = 'center'
  for (const x of cfg.ticksX) ctx.fillText(String(x), toX(x), height - 8)
  ctx.textAlign = 'left'
  ctx.fillText('loss', 4, 12)
  ctx.textAlign = 'right'
  ctx.fillText('step', width - pad.r, height - 8)

  for (const guide of cfg.guides || []) {
    ctx.strokeStyle = rgb(colors[guide.color])
    ctx.setLineDash([5, 4])
    ctx.beginPath()
    ctx.moveTo(pad.l, toY(guide.y)); ctx.lineTo(width - pad.r, toY(guide.y))
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = rgb(colors[guide.color])
    ctx.textAlign = 'right'
    ctx.fillText(guide.label, width - pad.r - 2, toY(guide.y) - 4)
  }
  if (cfg.marker != null) {
    ctx.strokeStyle = rgb(colors.ink)
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(toX(cfg.marker), pad.t); ctx.lineTo(toX(cfg.marker), height - pad.b)
    ctx.stroke()
  }
  for (const series of cfg.series) {
    const color = rgb(colors[series.color])
    ctx.strokeStyle = color
    ctx.fillStyle = color
    ctx.lineWidth = series.dots ? 1.2 : 2
    ctx.beginPath()
    series.points.forEach(([x, y], i) => (i ? ctx.lineTo(toX(x), toY(y)) : ctx.moveTo(toX(x), toY(y))))
    ctx.stroke()
    if (series.dots) {
      for (const [x, y] of series.points) {
        ctx.beginPath()
        ctx.arc(toX(x), toY(y), 3.5, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  }
  // 圖例
  let lx = pad.l + 8
  ctx.textAlign = 'left'
  for (const series of cfg.series) {
    ctx.fillStyle = rgb(colors[series.color])
    if (series.dots) {
      ctx.beginPath()
      ctx.arc(lx + 5, pad.t + 6, 3.5, 0, Math.PI * 2)
      ctx.fill()
    } else ctx.fillRect(lx, pad.t + 5, 12, 2.5)
    ctx.fillStyle = rgb(colors.ink)
    ctx.fillText(series.label, lx + 16, pad.t + 10)
    lx += 22 + ctx.measureText(series.label).width + 10
  }
}

// ───────────────────────────── 第六關：訓練 ─────────────────────────────

INIT[6] = () => {
  const LRS = CONFIG.trainLRs, TOTAL = 300, TARGET = CONFIG.trainTarget
  const chart = $('#l6-chart'), stat = $('#l6-stat'), samples = $('#l6-samples'), fb = $('#l6-fb')
  const lrSlider = $('#l6-lr'), lrOut = $('#l6-lr-out'), runButton = $('#l6-run')

  // 這一關用自己的小字表，從零開始訓練
  const tok = T.CharTokenizer.fromTexts(POEMS)
  const split = T.splitDocs(POEMS, 0.1, new T.RNG(1))
  const trainSet = new T.Dataset(split.train, tok), valSet = new T.Dataset(split.val, tok)
  const baseline = Math.log(tok.vocabSize)
  let run

  const fresh = () => {
    const rng = new T.RNG((Math.random() * 2 ** 31) | 0)
    const net = new T.GPT({ vocabSize: tok.vocabSize, blockSize: 25, nLayer: 1, nHead: 2, nEmbd: 32 }, rng)
    run = { rng, net, optimizer: new T.AdamW(net.parameters()), step: 0, smoothed: 0, train: [], val: [], running: false, ms: 0, passed: false }
    samples.replaceChildren(h('div', {}, h('span', { text: '尚未開始' }), h('em', { class: 'muted', text: '按下開始，看它從亂碼學到格式' })))
    lrSlider.disabled = false
    runButton.disabled = false
    runButton.textContent = `開始訓練 ${TOTAL} 步`
    feedback(fb, '', `這顆模型有 ${net.parameterCount().toLocaleString()} 個參數，全是亂數。`)
    draw()
  }

  const draw = () => {
    lrOut.textContent = String(LRS[Number(lrSlider.value)])
    drawChart(chart, {
      maxX: TOTAL, minY: 4.2, maxY: 7.4, ticksX: [0, 100, 200, 300],
      guides: [{ y: baseline, color: 'ink-soft', label: `亂猜 ${baseline.toFixed(2)}` }, { y: TARGET, color: 'cinnabar', label: `目標 ${TARGET}` }],
      series: [{ label: '訓練 loss', color: 'azurite', points: run.train }, { label: '驗證 loss', color: 'gamboge', points: run.val, dots: true }],
    })
    const lastVal = run.val.at(-1)
    stat.replaceChildren(
      h('span', {}, 'step ', h('b', { text: `${run.step} / ${TOTAL}` })),
      h('span', {}, '訓練 loss ', h('b', { text: run.step ? run.smoothed.toFixed(3) : '—' })),
      h('span', {}, '驗證 loss ', h('b', { text: lastVal ? lastVal[1].toFixed(3) : '—' })),
      h('span', {}, '速度 ', h('b', { text: run.ms ? `${run.ms.toFixed(0)} ms/step` : '—' })),
    )
  }

  const evaluate = () => {
    const val = T.estimateLoss(run.net, valSet, new T.RNG(7), 3, 16)
    run.val.push([run.step, val])
    const ids = T.generate(run.net, [tok.endToken], { maxNewTokens: 25, temperature: 0.8, topK: 40, stopToken: tok.endToken, rng: run.rng })
    if (run.step === 0) samples.replaceChildren()
    samples.prepend(h('div', {}, h('span', { text: `第 ${run.step} 步` }), tok.decode(ids) || '（空）'))
    if (val < TARGET && !run.passed) {
      run.passed = true
      feedback(fb, 'ok', `第 ${run.step} 步，驗證 loss ${val.toFixed(3)}，過關。模型在沒看過的詩上也預測得更準，表示它學到的是規律。`)
      pass(6)
    }
  }

  const loop = async () => {
    const schedule = { baseLR: LRS[Number(lrSlider.value)], warmupSteps: 20, totalSteps: TOTAL }
    lrSlider.disabled = true
    if (run.step === 0) evaluate()
    let started = performance.now(), counted = 0
    while (run.running && run.step < TOTAL) {
      const { loss } = T.trainStep(run.net, run.optimizer, trainSet.batch(run.rng, 8, 25), T.learningRate(run.step, schedule))
      if (!Number.isFinite(loss)) {
        run.running = false
        feedback(fb, 'no', 'loss 變成 NaN，訓練發散了。換一顆新模型，把學習率調低。')
        break
      }
      run.smoothed = run.step === 0 ? loss : 0.9 * run.smoothed + 0.1 * loss
      run.step++
      counted++
      run.train.push([run.step, run.smoothed])
      if (run.step % 25 === 0) {
        run.ms = (performance.now() - started) / counted
        evaluate()
        started = performance.now()
        counted = 0
      }
      if (run.step % 3 === 0) draw()
      await new Promise((resolve) => setTimeout(resolve, 0)) // 把控制權還給瀏覽器，畫面才不會卡住
    }
    run.running = false
    draw()
    if (run.step >= TOTAL) {
      runButton.disabled = true
      runButton.textContent = '已跑完 300 步'
      if (!run.passed) {
        const final = run.val.at(-1)[1]
        feedback(fb, 'no', final > baseline - 0.8
          ? `驗證 loss 停在 ${final.toFixed(3)}，幾乎沒學到東西。換一顆新模型，學習率換個數量級試試。`
          : `驗證 loss ${final.toFixed(3)}，還差一點。換一顆新模型，調整學習率再跑一次。`)
      }
    } else runButton.textContent = '繼續'
  }

  runButton.addEventListener('click', () => {
    if (run.running) {
      run.running = false
      return
    }
    run.running = true
    runButton.textContent = '暫停'
    loop()
  })
  $('#l6-reset').addEventListener('click', () => {
    if (run) run.running = false
    setTimeout(fresh, 30)
  })
  lrSlider.addEventListener('input', draw)

  // 大模型的訓練紀錄
  const history = CKPT.meta.history, curve = CKPT.meta.lossCurve
  const big = $('#l6-big'), slider = $('#l6-hist'), label = $('#l6-hist-label'), poem = $('#l6-hist-poem')
  const totalSteps = CKPT.meta.steps
  slider.max = String(history.length - 1)
  slider.value = String(history.length - 1)
  const drawBig = () => {
    const item = history[Number(slider.value)]
    label.textContent = `第 ${item.step} 步　訓練 loss ${item.train.toFixed(2)}　驗證 loss ${item.val.toFixed(2)}`
    poem.textContent = item.sample
    drawChart(big, {
      maxX: totalSteps, minY: 3.8, maxY: 7.8, ticksX: [0, 1000, 2000, 3000].filter((x) => x <= totalSteps), marker: item.step,
      guides: [{ y: Math.log(model.config.vocabSize), color: 'ink-soft', label: `亂猜 ${Math.log(model.config.vocabSize).toFixed(2)}` }],
      series: [
        { label: '訓練 loss', color: 'azurite', points: curve.map((y, i) => [i * 10, y]) },
        { label: '驗證 loss', color: 'gamboge', points: history.map((p) => [p.step, p.val]), dots: true },
      ],
    })
  }
  slider.addEventListener('input', drawBig)

  REDRAW[6] = () => {
    draw()
    drawBig()
  }
  fresh()
  drawBig()
}

// ───────────────────────────── 第七關：生成 ─────────────────────────────

INIT[7] = () => {
  const poem = $('#l7-poem'), bars = $('#l7-bars'), fb = $('#l7-fb'), trialFb = $('#l7-trial-fb')
  const prompt = $('#l7-prompt'), temp = $('#l7-temp'), topk = $('#l7-topk')
  const rng = new T.RNG((Math.random() * 2 ** 31) | 0)
  const MAX = model.config.blockSize - 1
  let ids, mine, done, logits

  const settings = () => ({ temperature: Number(temp.value) / 100, topK: Number(topk.value) })

  const predict = () => {
    logits = done ? null : T.noGrad(() => Float64Array.from(lastRow(model.forward(ids))))
  }

  const render = () => {
    const { temperature, topK: k } = settings()
    $('#l7-temp-out').textContent = temperature.toFixed(2) + (temperature === 0 ? '（永遠選第一名）' : '')
    $('#l7-topk-out').textContent = String(k)

    const spans = []
    ids.slice(1).forEach((id, i) => {
      spans.push(h('span', { class: mine.has(i + 1) ? 'mine' : null, text: tokenizer.tokenText(id) }))
      if (tokenizer.tokenText(id) === '。') spans.push(h('br'))
    })
    if (!done) spans.push(h('span', { class: 'cursor' }))
    poem.replaceChildren(...spans)

    if (done) {
      bars.replaceChildren(h('p', { class: 'muted', text: '寫完了。按「重寫」再來一首，或換個開頭、換個溫度。' }))
    } else {
      const ranked = topK(T.samplingDistribution(logits, { temperature, topK: k }), 10)
      bars.replaceChildren(...ranked.map(({ id, p }) => barRow(tokenLabel(id), p, ranked[0].p, { onclick: () => push(id, true) })))
    }
    for (const id of ['#l7-one', '#l7-all']) $(id).disabled = done
    const picked = mine.size
    if (done && picked >= 3) {
      feedback(trialFb, 'ok', `過關。這首詩有 ${picked} 個字是你選的，其餘是模型依機率抽的。同一個開頭，再寫一次會是另一首。`)
      pass(7)
    } else if (done) {
      feedback(trialFb, 'no', `這首只有 ${picked} 個字是你親手選的。按「重寫」，過程中點機率長條至少三次。`)
    } else {
      feedback(trialFb, '', `已親手選 ${picked} 個字，目前寫了 ${ids.length - 1} 個字。`)
    }
  }

  const push = (id, byUser) => {
    if (done) return
    if (id === END) done = true
    else {
      ids.push(id)
      if (byUser) mine.add(ids.length - 1)
      if (ids.length - 1 >= MAX) done = true
    }
    predict()
    render()
  }

  const sampleOne = () => push(rng.multinomial(T.samplingDistribution(logits, settings())), false)

  const reset = () => {
    const text = prompt.value.trim()
    if (!canEncode(text)) {
      const bad = Array.from(text).find((ch) => !canEncode(ch))
      return feedback(fb, 'no', `字表裡沒有「${bad}」，換一個字。`)
    }
    feedback(fb, '', '')
    ids = [END, ...tokenizer.encode(text)]
    mine = new Set()
    done = false
    predict()
    render()
  }

  $('#l7-one').addEventListener('click', sampleOne)
  $('#l7-all').addEventListener('click', () => {
    const tick = () => {
      if (done) return
      sampleOne()
      setTimeout(tick, 70)
    }
    tick()
  })
  $('#l7-reset').addEventListener('click', reset)
  prompt.addEventListener('input', reset)
  temp.addEventListener('input', render)
  topk.addEventListener('input', render)
  reset()
}

// ───────────────────────────── 結語 ─────────────────────────────

REDRAW[8] = () => {
  const count = earned.filter(Boolean).length
  $('#fin-seals').replaceChildren(...SEALS.map((ch, i) => h('span', { class: 'seal' + (earned[i] ? ' earned' : ''), text: ch })))
  $('#fin-title').textContent = count === 7 ? '七個印都蓋齊了' : '結語'
  $('#fin-lead').textContent = count === 7
    ? '你親手做過斷字、反向傳播、梯度下降，看過 attention 的分工，組過 Transformer，訓練過一顆模型，還和它合寫了一首詩。'
    : `目前蓋了 ${count} 個印，還有 ${7 - count} 關的試煉沒完成。點上方的印章隨時可以回去。`
}

// ───────────────────────────── 啟動 ─────────────────────────────

const start = HASHES.indexOf(location.hash.slice(1))
show(start >= 0 ? start + 1 : 1)
})()
