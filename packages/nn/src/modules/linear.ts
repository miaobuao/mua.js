import { type Tensor, ones } from '@mua/tensor'

import { Module } from '.'

export class Linear extends Module {
  readonly weight: Tensor

  constructor(
    readonly inSize: number,
    readonly outSize: number,
  ) {
    super()
    this.weight = ones(inSize, outSize)
  }

  async forward(x: Tensor, y: Tensor) {
    const z = await x.mul(y)
    return z
  }
}
