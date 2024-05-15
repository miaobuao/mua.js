import type { MaybePromise } from '@mua/common'

import { isNil, range } from 'lodash-es'
import { NdArray, dtype } from 'ndarray-js'

import { TensorValueIsNullError } from './errors'
import { Graph, reshapeArray } from './helper'
import { getConfig } from './mode'
import { type OpTrait, add, addScalar, dot, matmul } from './ops'
import { mulScalar } from './ops/mul-scalar'

type Many<T> = T | Many<T>[]
const config = getConfig()

export class Tensor<Type extends dtype = dtype> {
  op: OpTrait | undefined
  inputs: Tensor[] = []
  private cache: NdArray<Type> | null = null
  requiresGrad = true
  gradient: Tensor | null = null
  readonly dtype: Type

  constructor(
    data?: NdArray<Type> | null | ArrayLike<Many<number>>,
    opts: {
      requiresGrad?: boolean
      dtype?: Type
    } = {},
  ) {
    const { requiresGrad, dtype: _dtype } = opts
    this.dtype = _dtype || dtype.float32 as Type
    if (!isNil(requiresGrad))
      this.requiresGrad = requiresGrad
    if (data instanceof NdArray)
      this.cache = data
    else if (!isNil(data))
      this.cache = new NdArray(data, { dtype: this.dtype })
  }

  static async fromOp(op: OpTrait, ...inputs: MaybePromise<Tensor>[]) {
    const t = new Tensor()
    t.op = op
    t.inputs = await Promise.all(inputs)
    if (!config.LAZY_MODE) {
      if (!config.REQUIRES_GRAD)
        return t.detach()
      await t.realize()
    }
    return t
  }

  async backward(outGrad: Tensor | null = null) {
    outGrad ??= await onesLike(this)
    await computeGradient(this, outGrad)
  }

  add(t: Tensor | number) {
    if (typeof t === 'number')
      return addScalar(this, t)
    return add(this, t)
  }

  matmul(t: Tensor) {
    return matmul(this, t)
  }

  dot(t: Tensor) {
    return dot(this, t)
  }

  mul(t: number | Tensor) {
    if (typeof t === 'number')
      return mulScalar(this, t)
    return this.dot(t)
  }

  async detach() {
    return new Tensor(
      await this.realize().then(d => d.raw),
      {
        requiresGrad: false,
      },
    )
  }

  private async realize() {
    if (isNil(this.cache)) {
      if (!this.op)
        throw new Error('op is null')
      const data = await this.op.compute(...this.inputs)
      if (isNil(data))
        throw new Error('data is null')
      if (data instanceof NdArray)
        this.cache = data as any
      else this.cache = new NdArray(data as any)
      return this
    }
    return this
  }

  get value() {
    return this.detach()
  }

  get raw(): Promise< NdArray<Type> | null> {
    return this.realize().then(d => d.cache)
  }

  get T() {
    return this.transpose()
  }

  async transpose() {
    const order = range((await this.shape).length).reverse()
    return this.permute(order)
  }

  async permute(order: number[]) {
    const raw = await this.raw
    if (!raw)
      throw new TensorValueIsNullError()
    return new Tensor(await raw.permute(order))
  }

  async reshape(size: number[]) {
    const raw = await this.raw
    if (!raw)
      throw new TensorValueIsNullError()
    return new Tensor(await raw.reshape(size), {
      requiresGrad: this.requiresGrad,
    })
  }

  async setRaw(v: MaybePromise<typeof this.cache>) {
    this.cache = await Promise.resolve(v)
  }

  async toArray() {
    const v = await this.raw
    if (v instanceof NdArray)
      return reshapeArray(Array.from(v.buffer), v.shape)
    throw new Error(`${v} cannot be converted to array`)
  }

  get shape() {
    return this.raw.then((v) => {
      if (v instanceof NdArray)
        return v.shape
      return [ 1 ]
    })
  }

  get buffer() {
    return this.raw.then(v =>
      v?.buffer,
    )
  }

  async sum() {
    const v = await this.raw
    if (v instanceof NdArray)
      return v.sum()
    throw new Error(`${v} cannot be summed`)
  }

  async argmax() {
    const value = await this.raw
    return value?.argmax()
  }
}

export function detach(t: MaybePromise<Tensor>) {
  return Promise.resolve(t).then(d => d.detach())
}

export function transpose(t: MaybePromise<Tensor>) {
  return Promise.resolve(t).then(d => d.T)
}

export function ones(size: number[]) {
  return new Tensor(NdArray.ones(size))
}

export async function onesLike(t: MaybePromise<Tensor>) {
  t = await t
  const shape = await t.shape
  return ones(shape)
}

export function zeros(size: number[]) {
  return new Tensor(NdArray.zeros(size))
}

export async function zerosLike(t: MaybePromise<Tensor>) {
  t = await t
  const shape = await t.shape
  return zeros(shape)
}

// export function random(size: number[], max: number): Tensor
// export function random(size: number[], min: number, max: number): Tensor
// export function random(size, min?: number, max?: number)
export function random(size: number[]): Tensor {
  // if (min === undefined)
  //   return new Tensor(_random(size))
  // else if (max === undefined)
  //   return new Tensor(_random(size, min))
  // else
  return new Tensor(NdArray.random((size)))
}

export function normal(size: number[], mean = 0, std = 0.01) {
  return new Tensor(NdArray.normal(size, { mean, std }))
}

export async function computeGradient(outNode: Tensor, outGrad: Tensor) {
  const node2Grad = new WeakMap([ [ outNode, [ outGrad ] ] ])
  const g = new Graph<Tensor>()
  const queues: Tensor[] = [ outNode ]
  while (queues.length) {
    const node = queues.pop()!
    node.inputs.forEach((input) => {
      g.addEdge(node, input)
      queues.push(input)
    })
  }
  // reverse topological sort
  for (const node of g.sort()) {
    const grad = await node2Grad
      .get(node)!
      .map(d => Promise.resolve(d))
      .reduce(add)
    if (grad)
      node.gradient = await grad.detach()

    if (!node.op)
      continue

    const inputGrads = await node.op!.gradient(grad, ...node.inputs)

    for (let i = 0; i < node.inputs.length; ++i) {
      const input = node.inputs[i]!
      const grads = node2Grad.get(input) || []
      grads.push(inputGrads[i]!)
      node2Grad.set(input, grads)
    }
  }
}
