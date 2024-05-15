import { type MaybePromise, asyncValueNotNil } from '@mua/common'
import { isNil } from 'lodash-es'
// import { dot } from 'ndarray-js'

import { OpTrait } from './op-trait'
import { Tensor } from '../tensor'

class LogOps extends OpTrait {
  constructor(readonly base: number | undefined = Math.E) { super() }

  async compute(x: MaybePromise<Tensor>) {
    const raw = await Promise.resolve(x).then(d => d.raw)
    return raw!.log(this.base!)
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]): Promise<[Tensor]> {
    const [ outGrad, input ] = await Promise.all([ grad, ...inputs ])
    let res: any
    const lhs = await asyncValueNotNil(outGrad.raw)
    if (isNil(this.base) || this.base === Math.E) {
      const rhs = await asyncValueNotNil(input.raw).then(d => d.pow(-1))
      res = await lhs.mul(rhs)
    }
    else {
      const logBase = Math.log(this.base!)
      const rhs = await asyncValueNotNil(input.raw).then(d => d.map((d) => {
        const bottom = logBase * d
        return bottom === 0 ? 1e2 : 1 / bottom
      }))
      res = lhs.mul(rhs)
    }
    return [ new Tensor(res) ]
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
