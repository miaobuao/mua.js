import { type MaybePromise, type Tensor, add } from '@mua/tensor'

import { Module } from '../modules'

export class L2Loss extends Module {
  async forward(x: MaybePromise<Tensor>, y: MaybePromise<Tensor>) {
    const [ t1, t2 ] = await Promise.all([ x, y ])
    const z = await add(t1, t2.mul(-1))
    return z.dot(z)
  }
}
