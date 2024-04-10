import type { Tensor } from '@mua/tensor'

import { Optimizer } from '.'

export class SGD extends Optimizer {
  constructor(
    params: Tensor[],
    public lr = 0.01,
  ) {
    super(params)
  }

  async step() {
    return Promise.all(this.params.map(async p => p.setRaw(
      (await p.value).add(
        await p.gradient!.mul(-this.lr),
      ).then(d => d.raw),
    )))
  }
}
