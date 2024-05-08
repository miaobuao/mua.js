import type { MaybePromise } from '@mua/common'

import { type Tensor, matmul, normal } from '@mua/tensor'

import { Module } from './module'

export class Linear extends Module {
  readonly weight: Tensor
  // readonly bias: Tensor

  constructor(
    readonly inSize: number,
    readonly outSize: number,
  ) {
    super()
    this.weight = normal([ inSize, outSize ])
    // this.bias = ones([ outSize ])
  }

  async forward(x: MaybePromise<Tensor>) {
    x = matmul(x, this.weight)
    // x = add(x, this.bias)
    return x
  }
}
