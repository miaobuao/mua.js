import type { MaybePromise } from '../helper'

import { softmax as _sm } from 'async-math'

import { OpTrait } from './op-trait'
import { Tensor } from '..'
import { TensorValueIsNullError } from '../errors'

export class Softmax extends OpTrait {
  async compute(a: Tensor) {
    const arr = await a.raw
    if (arr === null)
      throw new TensorValueIsNullError()
    return _sm(arr.value)
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]) {
    throw new Error('not implemented')
    const [ outGrad, input ] = await Promise.all([ grad, inputs[0] ])
    return [ outGrad ]
  }
}

export function softmax(a: MaybePromise<Tensor>) {
  const op = new Softmax()
  return Tensor.fromOp(op, a)
}
