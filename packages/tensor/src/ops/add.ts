import type { MaybePromise } from '../helper'

import { add as _add } from 'async-math'

import { OpTrait } from './op-trait'
import { Tensor } from '..'
import { TensorValueIsNullError } from '../errors'

export class Add extends OpTrait {
  async compute(a: Tensor, b: Tensor) {
    const v1 = await a.raw
    const v2 = await b.raw
    if (v1 === null || v2 === null)
      throw new TensorValueIsNullError()
    return _add(v1, v2)
  }

  async gradient(grad: Tensor, ...inputs: [Tensor, Tensor]) {
    if (inputs.length !== 2)
      throw new Error(`add: expected 2 input, got ${inputs.length}`)
    return Promise.all([
      grad.detach(),
      grad.detach(),
    ])
  }
}

export async function add(
  a: MaybePromise<Tensor>,
  b: MaybePromise<Tensor>,
) {
  const op = new Add()
  const [ t1, t2 ] = await Promise.all([ a, b ])
  return Tensor.fromOp(op, t1, t2)
}
