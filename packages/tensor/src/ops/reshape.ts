import type { MaybePromise } from '@mua/common'

import { OpTrait } from './op-trait'
import { Tensor } from '../tensor'

export class ReshapeOps extends OpTrait {
  readonly shape: Int32Array

  constructor(
    shape: number[],
  ) {
    super()
    this.shape = new Int32Array(shape)
  }

  async compute(x: Tensor) {
    const raw = await x.raw
    return raw!.reshape(this.shape)
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]): Promise<[Tensor]> {
    const [ outGrad, input ] = await Promise.all([ grad, ...inputs ])
    return [
      await outGrad.reshape(
        await input.shape,
      ),
    ]
  }
}

export function reshape(x: MaybePromise<Tensor>, shape: number[]) {
  const op = new ReshapeOps(shape)
  return Tensor.fromOp(op, x)
}
