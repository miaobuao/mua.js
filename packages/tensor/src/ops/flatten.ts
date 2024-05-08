import type { MaybePromise } from '@mua/common'

import { OpTrait } from './op-trait'
import { TensorValueIsNullError } from '../errors'
import { assert } from '../helper'
import { Tensor } from '../tensor'

class FlattenOps extends OpTrait {
  async compute(x: Tensor) {
    const v = await x.raw
    if (v === null)
      throw new TensorValueIsNullError()

    const shape = v.shape
    return v.reshape(new Int32Array([ 1, shape.reduce((a, b) => a * b) ]))
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]) {
    assert(inputs.length === 1, `flatten: expected 1 input, got ${inputs.length}`)
    const input = await Promise.resolve(inputs[0]).then(d => d.raw)
    grad = await grad
    return [ await grad.reshape(input!.shape) ]
  }
}

export function flatten(x: MaybePromise<Tensor>) {
  const op = new FlattenOps()
  return Tensor.fromOp(op, x)
}
