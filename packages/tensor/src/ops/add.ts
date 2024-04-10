import { add as _add } from 'async-math'

import { OpTrait } from '.'
import { Tensor } from '..'
import { TensorValueIsNullError, TensorValueTypeError } from '../errors'

export class Add extends OpTrait {
  async compute(a: Tensor, b: Tensor) {
    const v1 = await a.raw
    const v2 = await b.raw
    if (v1 === null || v2 === null)
      throw new TensorValueIsNullError()
    if (typeof v1 === 'number' || typeof v2 === 'number')
      throw new TensorValueTypeError()
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

export function add(a: Tensor, b: Tensor) {
  const op = new Add()
  return Tensor.fromOp(op, a, b)
}
