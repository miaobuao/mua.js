import type { MaybePromise } from '../helper'

import { Matrix, dot } from 'async-math'

import { OpTrait } from './op-trait'
import { Tensor, TensorValueTypeError } from '..'

export class ReLU extends OpTrait {
  async compute(x: Tensor) {
    return x.raw.then((d) => {
      if (d instanceof Matrix)
        return d.map(v => v > 0 ? v : 0)
      throw new TensorValueTypeError()
    })
  }

  async gradient(outGrad: Tensor, ...inputs: [Tensor]) {
    const [ grad, input ] = await Promise.all([
      outGrad.raw,
      inputs[0].raw,
    ])
    return [
      new Tensor(
        await dot(
          grad!,
          input!.map(v => v > 0 ? 1 : 0),
        ),
      ),
    ]
  }
}

export async function relu(x: MaybePromise<Tensor>) {
  const op = new ReLU()
  return Tensor.fromOp(op, await x)
}
