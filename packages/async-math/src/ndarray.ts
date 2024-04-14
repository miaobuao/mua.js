import { range, uniq } from 'lodash-es'
import { reshape, size, zeros } from 'mathjs'

export type NdArrayCell<T> = NdArrayCell<T>[] | T
export type NdArrayNumberCell = NdArrayCell<number>

export class NdArray<TValueType = number> {
  constructor(
    public readonly value: NdArrayCell<TValueType>[],
  ) {}

  toArray() {
    return this.value
  }

  static toValue<T>(t: T): T
  static toValue<T>(t: NdArray<T>): NdArrayCell<T>
  static toValue(t) {
    if (t instanceof NdArray)
      return t.value
    return t
  }

  concat(...arrays: NdArrayCell<TValueType>[]) {
    return new NdArray(this.value.concat(...arrays))
  }

  permute(order: number[]) {
    return permute(this.value, order)
  }

  get T() {
    const order = range(this.shape.length).reverse()
    return this.permute(order)
  }

  reshape(size: number[]) {
    return new NdArray(
      reshape(this.value as number[], size),
    )
  }

  get shape() {
    return size(this.value as number[]) as number[]
  }

  mapRow(fn: (d: NdArrayCell<TValueType>, i: number) => NdArrayCell<TValueType>) {
    return new NdArray(this.value.map(fn))
  }

  mapElement(fn: (d: TValueType, i: number[]) => number) {
    return new NdArray(mapElementForArray(this.value, fn))
  }
}

export async function permute<TValueType = unknown>(
  value: NdArrayCell<TValueType>[],
  order: number[],
) {
  const shape = size(value as number[]) as number[]
  checkAxisOrder(order, shape)

  const orderDict: Record<number, number> = {}
  for (let i = 0; i < order.length; ++i)
    orderDict[i] = order[i]!
  const res = zeros(
    shape.map((_, i) => shape[orderDict[i]!]!),
  ) as NdArrayCell<TValueType>[]

  forEachElementForArray(value, (d, i) => {
    const idx = range(shape.length).map(d => i[orderDict[d]!]!)
    setElementForArray(res, d, idx)
  })
  return new NdArray(res)
}

export function forEachElementForArray<TValueType = unknown>(
  arr: NdArrayCell<TValueType>[],
  fn: (d: TValueType, i: number[]) => any,
) {
  function dfs(cell: NdArrayCell<TValueType>[], i: number[]) {
    if (Array.isArray(cell))
      cell.forEach((cell, j) => dfs(cell as NdArrayCell<TValueType>[], i.concat(j)))
    else
      fn(cell, i)
  }
  dfs(arr, [])
}

export function mapElementForArray<TValueType = unknown, TFnReturnType = unknown>(
  arr: NdArrayCell<TValueType>[],
  fn: (d: TValueType, i: number[]) => TFnReturnType,
): NdArrayCell<TFnReturnType>[] {
  function dfs(cell: any, i: number[]) {
    if (Array.isArray(cell))
      return cell.map((cell, j) => dfs(cell, i.concat(j)))
    else
      return fn(cell, i)
  }
  return dfs(arr, [])
}

export function setElementForArray<TValueType = unknown>(
  arr: NdArrayCell<TValueType>[],
  value: NdArrayCell<TValueType>,
  index: number[],
) {
  const len = index.length
  let target: NdArrayCell<TValueType> = arr
  for (let i = 0; i < len - 1; ++i)
    target = target[index[i]!]!
  target[index[len - 1]!] = value
}

function checkAxisOrder(order: number[], shape: number[]) {
  if (order.length !== shape.length)
    throw new Error('axisOrder must be same length as shape')
  const uniqueOrder = uniq(order)
  if (uniqueOrder.length !== shape.length || uniqueOrder.some(d => !order.includes(d)))
    throw new Error(`wrong order: ${order}`)
}
