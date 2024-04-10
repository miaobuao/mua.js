import type { MaybePromise } from '../helper'

import { softmax as _sm } from 'async-math'

import { OpTrait } from './op-trait'
import { Tensor } from '..'
import { TensorValueIsNullError, TensorValueTypeError } from '../errors'

export class Softmax extends OpTrait {
  async compute(a: Tensor) {
    const arr = await a.raw
    if (arr === null)
      throw new TensorValueIsNullError()
    if (typeof arr === 'number')
      throw new TensorValueTypeError()
    return _sm(arr)
  }

  async gradient(grad: Tensor, ...inputs: [Tensor]): Promise<any> {
    return grad
  }
}

export async function softmax(a: MaybePromise<Tensor>) {
  const op = new Softmax()
  return Tensor.fromOp(op, await Promise.resolve(a))
}
