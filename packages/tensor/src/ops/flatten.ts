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
    return v.reshape([ 1, shape.reduce((a, b) => a * b) ])
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]) {
    assert(inputs.length === 1, `flatten: expected 1 input, got ${inputs.length}`)
    const input = await Promise.resolve(inputs[0]).then(d => d.raw)
    const outGrad = await Promise.resolve(grad).then(d => d.raw)
    return [ new Tensor(outGrad!.reshape(input!.shape)) ]
  }
}

export function flatten(x: MaybePromise<Tensor>) {
  const op = new FlattenOps()
  return Tensor.fromOp(op, x)
}
