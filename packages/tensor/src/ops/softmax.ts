import { matmul as _matmul } from 'async-math'

import { OpTrait } from '.'
import { Tensor } from '..'
import { TensorValueIsNullError, TensorValueTypeError } from '../errors'

export class Softmax extends OpTrait {
  async compute(a: Tensor) {
    const v1 = await a.raw
    if (v1 === null)
      throw new TensorValueIsNullError()
    if (typeof v1 === 'number')
      throw new TensorValueTypeError()
    return _matmul(v1, v1)
  }

  async gradient(grad: Tensor, inputs: Tensor[]): Promise<any> {
    return grad
  }
}

export function softmax(a: Tensor) {
  const op = new Softmax()
  return Tensor.fromOp(op, a)
}
