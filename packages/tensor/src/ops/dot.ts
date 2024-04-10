import { dot as _dot } from 'async-math'

import { OpTrait } from '.'
import { Tensor, detach } from '..'
import { TensorValueIsNullError, TensorValueTypeError } from '../errors'

export class Dot extends OpTrait {
  async compute(a: Tensor, b: Tensor) {
    const v1 = await a.raw
    const v2 = await b.raw
    if (v1 === null || v2 === null)
      throw new TensorValueIsNullError()
    if (typeof v1 === 'number' || typeof v2 === 'number')
      throw new TensorValueTypeError()
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

export function dot(a: Tensor, b: Tensor): Promise<Tensor> {
  const op = new Dot()
  return Tensor.fromOp(op, a, b)
}
