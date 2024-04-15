import type { MaybePromise } from '@mua/common'

import { NdArray, dot } from 'async-math'

import { OpTrait } from './op-trait'
import { Tensor, TensorValueTypeError } from '..'

export class ReLU extends OpTrait {
  constructor(private readonly leaky: number = 0) { super() }

  async compute(x: Tensor) {
    return x.raw.then((d) => {
      if (d instanceof NdArray)
        return d.mapElement(v => v > 0 ? v : this.leaky)
      throw new TensorValueTypeError()
    })
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]) {
    const [ outGrad, input ] = await Promise.all(
      [
        Promise.resolve(grad).then(d => d.raw),
        Promise.resolve(inputs[0]).then(d => d.raw),
      ],
    )
    const res = await dot(outGrad!.value, input!.mapElement(v => v > 0 ? 1 : 0).value)
    return [ new Tensor(res) ]
  }
}

export function relu(x: MaybePromise<Tensor>, leaky: number = 0) {
  const op = new ReLU(leaky)
  return Tensor.fromOp(op, x)
}
