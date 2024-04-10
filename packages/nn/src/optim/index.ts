import type { Tensor } from '@mua/tensor'

export abstract class Optimizer {
  constructor(
    readonly params: Tensor[],
  ) {}

  resetGrad() {
    for (let i = 0; i < this.params.length; ++i)
      this.params[i]!.gradient = null
  }

  abstract step()
}

export * from './sgd'
