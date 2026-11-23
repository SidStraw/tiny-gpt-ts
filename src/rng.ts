// 可重現的亂數產生器。
// 機器學習的實驗需要「同樣的種子 → 同樣的結果」才有辦法除錯，
// 而 Math.random() 無法指定種子，所以自己做一個（mulberry32 演算法）。
export class RNG {
  private state: number

  constructor(seed = 42) {
    this.state = seed >>> 0
  }

  /** [0, 1) 均勻分布 */
  random(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0
    let t = this.state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }

  /** 0 到 n-1 的整數 */
  randint(n: number): number {
    return Math.floor(this.random() * n)
  }

  /** 標準常態分布（Box-Muller 轉換：兩個均勻亂數 → 一個常態亂數） */
  randn(): number {
    const u = 1 - this.random()
    const v = this.random()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }

  /** 依機率分布抽一個索引：把 [0,1) 切成長度為 probs[i] 的區段，看亂數落在哪一段 */
  multinomial(probs: ArrayLike<number>): number {
    let r = this.random()
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i]
      if (r < 0) return i
    }
    return probs.length - 1
  }

  /** Fisher-Yates 洗牌（原地） */
  shuffle<T>(items: T[]): T[] {
    for (let i = items.length - 1; i > 0; i--) {
      const j = this.randint(i + 1)
      ;[items[i], items[j]] = [items[j], items[i]]
    }
    return items
  }
}
