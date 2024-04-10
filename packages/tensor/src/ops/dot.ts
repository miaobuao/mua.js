import type { MaybePromise } from '../helper'

import { dot as _dot } from 'async-math'

import { OpTrait } from './op-trait'
import { Tensor, detach } from '..'
import { TensorValueIsNullError } from '../errors'

export class Dot extends OpTrait {
  async compute(a: Tensor, b: Tensor) {
    const v1 = await a.raw
    const v2 = await b.raw
    if (v1 === null || v2 === null)
      throw new TensorValueIsNullError()
    return _dot(v1, v2)
  }

  async gradient(grad: Tensor, ...inputs: [Tensor, Tensor]) {
    const [ l, r ] = inputs
    return Promise.all([
      detach(grad.dot(r)),
      detach(grad.dot(l)),
    ])
  }
}

export async function dot(
  a: MaybePromise<Tensor>,
  b: MaybePromise<Tensor>,
): Promise<Tensor> {
  const op = new Dot()
  const [ t1, t2 ] = await Promise.all([ a, b ])
  return Tensor.fromOp(op, t1, t2)
}
