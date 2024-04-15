import type { Tensor } from '@mua/tensor'

import { add } from '@mua/tensor'

import { Optimizer } from './optimizer'

export class SGD extends Optimizer {
  constructor(
    params: Tensor[],
    public lr = 0.01,
  ) {
    super(params)
  }

  async step() {
    return Promise.all(
      this.params.map(
        (p) => {
          return p.setRaw(
            add(
              p,
              p.gradient!.mul(-this.lr),
            ).then(d => d.raw),
          )
        },
      ),
    )
  }
}
