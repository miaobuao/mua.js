import type { MaybePromise } from '@mua/common'

import { type Tensor, add, matmul, normal, ones } from '@mua/tensor'

import { Module } from './module'

export class Linear extends Module {
  readonly weight: Tensor
  readonly bias: Tensor

  constructor(
    readonly inSize: number,
    readonly outSize: number,
  ) {
    super()
    this.weight = normal([ inSize, outSize ])
    this.bias = ones(outSize)
  }

  forward(x: MaybePromise<Tensor>) {
    return add(matmul(x, this.weight), this.bias)
  }
}
