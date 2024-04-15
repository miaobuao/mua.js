import type { MaybePromise } from '@mua/common'
import type { NdArrayCell } from 'async-math'

import { NdArray, normal as _normal, ones as _ones, random as _random, sum as _sum, zeros as _zeros } from 'async-math'
import { isNil, range } from 'lodash-es'

import { TensorValueIsNullError } from './errors'
import { Graph } from './helper'
import { getConfig } from './mode'
import { type OpTrait, add, addScalar, dot, matmul } from './ops'
import { mulScalar } from './ops/mul-scalar'

const config = getConfig()

export class Tensor<TDtype = any> {
  op: OpTrait | undefined
  inputs: Tensor<TDtype>[] = []
  private cache: NdArray<TDtype> | null = null
  requiresGrad = true
  gradient: Tensor | null = null

  constructor(data?: NdArray<TDtype> | NdArrayCell<TDtype>[] | null, opts: {
    requiresGrad?: boolean
  } = {}) {
    const { requiresGrad } = opts
    if (!isNil(requiresGrad))
      this.requiresGrad = requiresGrad

    if (data instanceof NdArray)
      this.cache = data
    else if (!isNil(data))
      this.cache = new NdArray(data)
  }

  static async fromOp(op: OpTrait, ...inputs: MaybePromise<Tensor>[]) {
    const t = new Tensor()
    t.op = op
    t.inputs = await Promise.all (inputs)
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
    return new Tensor(await (await this.realize()).raw, {
      requiresGrad: false,
    })
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
      else
        this.cache = new NdArray(data) as any
      return this
    }
    return this
  }

  get value() {
    return this.detach()
  }

  get raw(): Promise<typeof this.cache> {
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
    return new Tensor(
      await raw.reshape(size),
      { requiresGrad: this.requiresGrad },
    )
  }

  async setRaw(v: MaybePromise<typeof this.cache>) {
    this.cache = await Promise.resolve(v)
  }

  async toArray() {
    const v = await this.raw
    if (v instanceof NdArray)
      return v.value
    throw new Error(`${v} cannot be converted to array`)
  }

  get shape() {
    return this.raw.then((v) => {
      if (v instanceof NdArray)
        return v.shape
      return [ 1 ]
    })
  }

  async sum(dim?: number) {
    const v = await this.raw
    if (v instanceof NdArray) {
      const data = v.value as number[]
      return dim === undefined
        ? _sum(data as number[])
        : _sum(data as number[], dim)
    }
    throw new Error(`${v} cannot be summed`)
  }
}

export function detach(t: MaybePromise<Tensor>) {
  return Promise.resolve(t).then(d => d.detach())
}

export function transpose(t: MaybePromise<Tensor>) {
  return Promise.resolve(t).then(d => d.T)
}

export function ones(...size: number[]) {
  return new Tensor(_ones(size) as number[])
}

export async function onesLike(t: MaybePromise<Tensor>) {
  t = await t
  const shape = await t.shape
  return ones(...shape)
}

export function zeros(...size: number[]) {
  return new Tensor(_zeros(size) as number[])
}

export async function zerosLike(t: MaybePromise<Tensor>) {
  t = await t
  const shape = await t.shape
  return zeros(...shape)
}

export function random(size: number[]): Tensor
export function random(size: number[], max: number): Tensor
export function random(size: number[], min: number, max: number): Tensor
export function random(size, min?: number, max?: number) {
  if (min === undefined)
    return new Tensor(_random(size))
  else if (max === undefined)
    return new Tensor(_random(size, min))
  else
    return new Tensor(_random(size, min, max))
}

export function normal(size: number[], mean = 0, std = 0.01) {
  return new Tensor(_normal(size, mean, std) as number[])
}

export function randomLike(t: MaybePromise<Tensor>): Promise<Tensor>
export function randomLike(t: MaybePromise<Tensor>, max: number): Promise<Tensor>
export function randomLike(t: MaybePromise<Tensor>, min: number, max: number): Promise<Tensor>
export async function randomLike(t: MaybePromise<Tensor>, min?: number, max?: number) {
  t = await t
  const shape = await t.shape
  if (min === undefined)
    return random(shape)
  else if (max === undefined)
    return random(shape, min)
  else
    return random(shape, min, max)
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
    const grad = await (node2Grad.get(node)!).map(d => Promise.resolve(d)).reduce(add)
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
