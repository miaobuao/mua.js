import type { MaybePromise } from './helper'
import type { MathCollection } from 'async-math'

import { Matrix, matrix } from 'async-math'
import { isNil, isNumber } from 'lodash-es'

import { getConfig } from './mode'
import { type OpTrait, add, addScalar, dot, matmul } from './ops'
import { mulScalar } from './ops/mul-scalar'

const config = getConfig()

export class Tensor {
  op: OpTrait | undefined
  inputs: Tensor[] = []
  private cache: MathCollection | number | null = null
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
    return new Tensor(await this.realize(), {
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
      return data
    }
    return this.cache
  }

  get value() {
    return this.detach()
  }

  get raw() {
    return this.realize()
  }

  async setRaw(v: MaybePromise<MathCollection>) {
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
}

export class Parameter extends Tensor {}
