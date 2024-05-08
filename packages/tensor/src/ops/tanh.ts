import type { MaybePromise } from '@mua/common'
import type { NdArray } from 'ndarray'

import { OpTrait } from './op-trait'
import { Tensor } from '../tensor'

export class TanhOp extends OpTrait {
  private ir: NdArray | null = null

  async compute(x: MaybePromise<Tensor>): Promise<NdArray> {
    x = await x
    const res = await x.raw.then(d => d!.tanh())
    this.ir = res
    return res
  }

  async gradient(grad: MaybePromise<Tensor>) {
    const outGrad = await grad
    const res = this.ir?.pow(2)
      .mulScalar(-1).addScalar(1).dot(
        (await outGrad.raw)!,
      )
    return [ new Tensor(res) ]
  }
}

export async function tanh(x: MaybePromise<Tensor>) {
  const op = new TanhOp()
  return Tensor.fromOp(op, x)
}
