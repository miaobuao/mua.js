import type { MathCollection } from 'async-math'

import { Matrix, ones as _ones, sum as _sum, transpose as _t, zeros as _zeros, matrix } from 'async-math'
import { isNil, isNumber } from 'lodash-es'

import { TensorValueIsNullError, TensorValueTypeError } from './errors'
import { Graph, type MaybePromise } from './helper'
import { getConfig } from './mode'
import { type OpTrait, add, addScalar, dot, matmul } from './ops'
import { mulScalar } from './ops/mul-scalar'

const config = getConfig()

export class Tensor {
  op: OpTrait | undefined
  inputs: Tensor[] = []
  private cache: Matrix | number | null = null
  private requiresGrad = true
  gradient: Tensor | null = null

  constructor(data?: MathCollection | number | null, opts: {
    requiresGrad?: boolean
  } = {}) {
    const { requiresGrad } = opts
    if (!isNil(requiresGrad))
      this.requiresGrad = requiresGrad

    if (data instanceof Matrix || isNumber(data))
      this.cache = data
    else if (!isNil(data))
      this.cache = matrix(data)
  }

  static async fromOp(op: OpTrait, ...inputs: Tensor[]) {
    const t = new Tensor()
    t.op = op
    t.inputs = inputs
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
      this.cache = data
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
    const value = await this.raw
    if (value === null)
      throw new TensorValueIsNullError()
    if (typeof value === 'number')
      throw new TensorValueTypeError()
    return new Tensor(_t(value))
  }

  async setRaw(v: MaybePromise<typeof this.cache>) {
    this.cache = await Promise.resolve(v)
  }

  async toArray() {
    const v = await this.raw
    if (v instanceof Matrix)
      return v.toArray()
    throw new Error(`${v} cannot be converted to array`)
  }

  get shape() {
    return this.raw.then((v) => {
      if (v instanceof Matrix)
        return v.size()
      return [ 1 ]
    })
  }

  async sum(dim?: number) {
    const v = await this.raw
    if (v instanceof Matrix)
      return dim === undefined ? _sum(v) : _sum(v, dim)
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
  return new Tensor(_ones(size))
}

export async function onesLike(t: MaybePromise<Tensor>) {
  t = await t
  const shape = await t.shape
  return ones(...shape)
}

export function zeros(...size: number[]) {
  return new Tensor(_zeros(size))
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
    node.gradient = grad

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
