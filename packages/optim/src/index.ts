import type { Parameter } from '@repo/tensor'

export abstract class Optimizer {
  constructor(readonly params: Parameter[]) {}

  resetGrad() {
    this.params.forEach(p => p.gradient = null)
  }

  abstract step()
}
