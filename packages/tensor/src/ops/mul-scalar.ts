import type { MaybePromise } from '../helper'

import { mulScalar as ms } from 'async-math'

import { OpTrait } from './op-trait'
import { Tensor, detach } from '..'
import { TensorValueIsNullError } from '../errors'

export class MulScalar extends OpTrait {
  constructor(readonly scalar: number) { super() }

  async compute(a: Tensor) {
    const v1 = await a.raw
    if (v1 === null)
      throw new TensorValueIsNullError()
    return ms(v1.value, this.scalar)
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]): Promise<any> {
    const outGrad = await grad
    return [
      await detach(
        outGrad.mul(this.scalar),
      ),
    ]
  }
}

export function mulScalar(a: MaybePromise<Tensor>, b: number) {
  const op = new MulScalar(b)
  return Tensor.fromOp(op, a)
}
