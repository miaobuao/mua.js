import type { MaybePromise } from '@mua/common'

import { OpTrait } from './op-trait'
import { Tensor, detach } from '..'
import { TensorValueIsNullError } from '../errors'

export class Dot extends OpTrait {
  async compute(a: Tensor, b: Tensor) {
    const v1 = await a.raw
    const v2 = await b.raw
    if (v1 === null || v2 === null)
      throw new TensorValueIsNullError()
    return v1.dot(v2)
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>, MaybePromise<Tensor>]) {
    const [ outGrad, l, r ] = await Promise.all([ grad, ...inputs ])
    return Promise.all([
      detach(outGrad.dot(r)),
      detach(outGrad.dot(l)),
    ])
  }
}

export async function dot(
  a: MaybePromise<Tensor>,
  b: MaybePromise<Tensor>,
): Promise<Tensor> {
  const op = new Dot()
  return Tensor.fromOp(op, a, b)
}
