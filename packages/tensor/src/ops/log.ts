import { type MaybePromise, asyncValueNotNil } from '@mua/common'
import { isNil } from 'lodash-es'
import { dot } from 'ndarray'

import { OpTrait } from './op-trait'
import { assert } from '../helper'
import { Tensor } from '../tensor'

class LogOps extends OpTrait {
  constructor(readonly base: number | undefined = undefined) { super() }

  async compute(x: MaybePromise<Tensor>) {
    const raw = await Promise.resolve(x).then(d => d.raw)
    assert(!isNil(raw), `log: tensor value is null`)
    return raw!.log(this.base!)
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]): Promise<[Tensor]> {
    const [ outGrad, input ] = await Promise.all([ grad, ...inputs ])
    if (isNil(this.base) || this.base === Math.E) {
      return [
        new Tensor(
          dot(
            await asyncValueNotNil(outGrad.raw),
            await asyncValueNotNil(input.raw).then(d => d.pow(-1)),
          ),
        ),
      ]
    }
    const logBase = Math.log(this.base!)
    return [
      new Tensor(
        dot(
          await asyncValueNotNil(outGrad.raw),
          await asyncValueNotNil(input.raw).then(d => d.map((d) => {
            const bottom = logBase * d
            return bottom === 0 ? 1e2 : 1 / bottom
          })),
        ),
      ),
    ]
  }
}

/**
 * @param x
 * @param base default `e`
 */
export function log(x: MaybePromise<Tensor>, base?: number) {
  const op = new LogOps(base)
  return Tensor.fromOp(op, x)
}
