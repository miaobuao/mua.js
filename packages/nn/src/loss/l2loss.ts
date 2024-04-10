import type { Tensor } from '@mua/tensor'

import { Module } from '../modules'

export class L2Loss extends Module {
  async forward(x: Tensor, y: Tensor) {
    const z = await x.add(await y.mul(-1))
    return z.dot(z)
  }
}
