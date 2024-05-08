import type { MaybePromise } from '@mua/common'

import { OpTrait } from './op-trait'
import { Tensor } from '..'
import { TensorValueIsNullError } from '../errors'

export class AddScalarOps extends OpTrait {
  constructor(readonly scalar: number) { super() };

  async compute(t: Tensor) {
    const v = await t.raw
    if (v === null)
      throw new TensorValueIsNullError()
    return v.addScalar(this.scalar)
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]) {
    if (inputs.length !== 1)
      throw new Error(`addScalar: expected 1 input, got ${inputs.length}`)
    const outGrad = await grad
    return [ await outGrad.detach() ]
  }
}

export function addScalar(t: MaybePromise<Tensor>, n: number) {
  const op = new AddScalarOps(n)
  return Tensor.fromOp(op, t)
}
