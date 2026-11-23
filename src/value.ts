// 第一站：純量（scalar）自動微分引擎。
//
// 這個檔案不會被 GPT 模型用到，它是「觀念版」：每個 Value 只裝一個數字，
// 用最少的程式碼展示 autograd 的全部精神。tensor.ts 是同一套想法的「矩陣版」。
//
// 核心想法：
//   1. 每做一次運算，就產生一個新的 Value，並記住「我是由誰算出來的」（parents）。
//      所有 Value 串起來就是一張計算圖（computation graph）。
//   2. 每個運算都知道自己的「局部導數」：輸出對各個輸入的微分。
//   3. backward() 從最終輸出往回走，用連鎖律把局部導數一路乘回去，
//      就得到最終輸出對每個 Value 的導數（grad）。
export class Value {
  data: number
  /** d(最終輸出) / d(這個值)，呼叫 backward() 之後才有意義 */
  grad = 0
  readonly parents: Value[]
  readonly op: string
  /** 把自己的 grad 依局部導數分配給 parents */
  backwardFn: () => void = () => {}

  constructor(data: number, parents: Value[] = [], op = '') {
    this.data = data
    this.parents = parents
    this.op = op
  }

  add(other: Value | number): Value {
    const o = toValue(other)
    const out = new Value(this.data + o.data, [this, o], '+')
    out.backwardFn = () => {
      // d(a+b)/da = 1、d(a+b)/db = 1：加法把梯度原封不動傳給兩邊
      this.grad += out.grad
      o.grad += out.grad
    }
    return out
  }

  mul(other: Value | number): Value {
    const o = toValue(other)
    const out = new Value(this.data * o.data, [this, o], '*')
    out.backwardFn = () => {
      // d(a*b)/da = b、d(a*b)/db = a：乘法把梯度乘上「對方的值」
      this.grad += o.data * out.grad
      o.grad += this.data * out.grad
    }
    return out
  }

  pow(exponent: number): Value {
    const out = new Value(this.data ** exponent, [this], `^${exponent}`)
    out.backwardFn = () => {
      this.grad += exponent * this.data ** (exponent - 1) * out.grad
    }
    return out
  }

  exp(): Value {
    const out = new Value(Math.exp(this.data), [this], 'exp')
    out.backwardFn = () => {
      this.grad += out.data * out.grad
    }
    return out
  }

  log(): Value {
    const out = new Value(Math.log(this.data), [this], 'log')
    out.backwardFn = () => {
      this.grad += (1 / this.data) * out.grad
    }
    return out
  }

  tanh(): Value {
    const t = Math.tanh(this.data)
    const out = new Value(t, [this], 'tanh')
    out.backwardFn = () => {
      this.grad += (1 - t * t) * out.grad
    }
    return out
  }

  relu(): Value {
    const out = new Value(Math.max(0, this.data), [this], 'relu')
    out.backwardFn = () => {
      this.grad += (this.data > 0 ? 1 : 0) * out.grad
    }
    return out
  }

  // 減法、除法不需要新的求導規則，用既有運算組合即可
  neg(): Value {
    return this.mul(-1)
  }
  sub(other: Value | number): Value {
    return this.add(toValue(other).neg())
  }
  div(other: Value | number): Value {
    return this.mul(toValue(other).pow(-1))
  }

  /**
   * 反向傳播。
   * 一個節點必須等「所有用到它的節點」都把梯度傳回來之後，才能往自己的 parents 傳。
   * 拓撲排序（topological sort）正好保證這個順序。
   */
  backward(): void {
    const order: Value[] = []
    const visited = new Set<Value>()
    const visit = (v: Value): void => {
      if (visited.has(v)) return
      visited.add(v)
      for (const p of v.parents) visit(p)
      order.push(v) // parents 全部排進去之後才輪到自己
    }
    visit(this)

    this.grad = 1 // d(自己)/d(自己) = 1
    for (let i = order.length - 1; i >= 0; i--) order[i].backwardFn()
  }
}

function toValue(x: Value | number): Value {
  return x instanceof Value ? x : new Value(x)
}
