import { matmul as _matmul } from 'async-math'

import { OpTrait } from '.'
import { Tensor } from '..'
import { TensorValueIsNullError, TensorValueTypeError } from '../errors'

export class MatMul extends OpTrait {
  async compute(a: Tensor, b: Tensor) {
    const v1 = await a.raw
    const v2 = await b.raw
    if (v1 === null || v2 === null)
      throw new TensorValueIsNullError()
    if (typeof v1 === 'number' || typeof v2 === 'number')
      throw new TensorValueTypeError()
    return _matmul(v1, v2)
  }

  async gradient(grad: Tensor, inputs: Tensor[]): Promise<any> {
    return grad
  }
}

export function matmul(a: Tensor, b: Tensor) {
  const op = new MatMul()
  return Tensor.fromOp(op, a, b)
}
