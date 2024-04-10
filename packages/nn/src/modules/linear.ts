import { type MaybePromise, type Tensor, matmul, ones } from '@mua/tensor'

import { Module } from './module'

export class Linear extends Module {
  readonly weight: Tensor

  constructor(
    readonly inSize: number,
    readonly outSize: number,
  ) {
    super()
    this.weight = ones(inSize, outSize)
  }

  forward(x: MaybePromise<Tensor>) {
    return matmul(x, this.weight)
  }
}
