import { type MaybePromise, type Tensor, add, mulScalar } from '@mua/tensor'

import { Module } from '../modules'

export class L2Loss extends Module {
  async forward(x: MaybePromise<Tensor>, y: MaybePromise<Tensor>) {
    const z = await add(x, mulScalar(y, -1))
    return z.dot(z)
  }
}
