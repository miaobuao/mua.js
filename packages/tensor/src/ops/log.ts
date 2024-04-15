import { type MaybePromise, asyncValueNotNil } from '@mua/common'
import { log as _log, dot } from 'async-math'
import { isNil } from 'lodash-es'

import { OpTrait } from './op-trait'
import { assert } from '../helper'
import { Tensor } from '../tensor'

class LogOps extends OpTrait {
  constructor(readonly base: number | undefined = undefined) { super() }

  async compute(x: MaybePromise<Tensor>) {
    const raw = await Promise.resolve(x).then(d => d.raw)
    assert(!isNil(raw), `log: tensor value is null`)
    return _log(raw!, this.base)
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]): Promise<[Tensor]> {
    const [ outGrad, input ] = await Promise.all([ grad, ...inputs ])
    if (isNil(this.base) || this.base === Math.E) {
      return [
        new Tensor(
          await dot(
            asyncValueNotNil(outGrad.raw),
            asyncValueNotNil(input.raw).then(d => d.mapElement(d => d === 0 ? 1e5 : 1 / d)),
          ),
        ),
      ]
    }
    return [
      new Tensor(
        await dot(
          asyncValueNotNil(outGrad.raw),
          asyncValueNotNil(input.raw).then(d => d.mapElement((d) => {
            const bottom = Math.log(this.base!) * d
            return bottom === 0 ? 1e5 : 1 / bottom
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
