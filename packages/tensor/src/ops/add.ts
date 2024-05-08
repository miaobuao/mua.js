import type { MaybePromise } from '@mua/common'

import { OpTrait } from './op-trait'
import { Tensor } from '..'
import { TensorValueIsNullError } from '../errors'

export class AddOps extends OpTrait {
  async compute(a: Tensor, b: Tensor) {
    const v1 = await a.raw
    const v2 = await b.raw
    if (v1 === null || v2 === null)
      throw new TensorValueIsNullError()
    return v1.add(v2)
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>, MaybePromise<Tensor>]) {
    if (inputs.length !== 2)
      throw new Error(`add: expected 2 input, got ${inputs.length}`)
    const outGrad = await grad

    return Promise.all([
      outGrad.detach(),
      outGrad.detach(),
    ])
  }
}

export function add(
  a: MaybePromise<Tensor>,
  b: MaybePromise<Tensor>,
) {
  const op = new AddOps()
  return Tensor.fromOp(op, a, b)
}
