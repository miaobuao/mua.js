import type { MaybePromise } from '@mua/common'

import { OpTrait } from './op-trait'
import { Tensor, detach } from '..'
import { TensorValueIsNullError } from '../errors'

export class MulScalar extends OpTrait {
  constructor(readonly scalar: number) { super() }

  async compute(a: MaybePromise<Tensor>) {
    const v1 = await (await a).raw
    if (v1 === null)
      throw new TensorValueIsNullError()
    return v1.mul(this.scalar)
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]): Promise<any> {
    const outGrad = await grad
    return [
      await detach(
        outGrad.mul(this.scalar),
      ),
    ]
  }
}

export function mulScalar(a: MaybePromise<Tensor>, b: number) {
  const op = new MulScalar(b)
  return Tensor.fromOp(op, a)
}
