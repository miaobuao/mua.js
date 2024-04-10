import { mulScalar as ms } from 'async-math'

import { OpTrait } from '.'
import { Tensor, detach } from '..'
import { TensorValueIsNullError, TensorValueTypeError } from '../errors'

export class MulScalar extends OpTrait {
  constructor(readonly scalar: number) { super() }

  async compute(a: Tensor) {
    const v1 = await a.raw
    if (v1 === null)
      throw new TensorValueIsNullError()
    if (typeof v1 === 'number')
      throw new TensorValueTypeError()
    return ms(v1, this.scalar)
  }

  async gradient(grad: Tensor, ...inputs: [Tensor]): Promise<any> {
    return [ await detach(grad.mul(this.scalar)) ]
  }
}

export function mulScalar(a: Tensor, b: number) {
  const op = new MulScalar(b)
  return Tensor.fromOp(op, a)
}
