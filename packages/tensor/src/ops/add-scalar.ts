import { addScalar as _addScalar } from 'async-math'
import { isNumber } from 'lodash-es'

import { OpTrait } from '.'
import { Tensor } from '..'
import { TensorValueIsNullError, TensorValueTypeError } from '../errors'

export class AddScalar extends OpTrait {
  constructor(readonly scalar: number) { super() };

  async compute(t: Tensor) {
    const v = await t.raw
    if (v === null)
      throw new TensorValueIsNullError()
    if (isNumber(v))
      throw new TensorValueTypeError()
    return _addScalar(v, this.scalar)
  }

  async gradient(grad: Tensor, inputs: Tensor[]): Promise<[Tensor]> {
    if (inputs.length !== 1)
      throw new Error(`addScalar: expected 1 input, got ${inputs.length}`)
    return [ grad ]
  }
}

export function addScalar(t: Tensor, n: number) {
  const op = new AddScalar(n)

  return Tensor.fromOp(op, t)
}
