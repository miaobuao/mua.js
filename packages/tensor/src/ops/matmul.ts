import type { MaybePromise } from '@mua/common'

import { OpTrait } from './op-trait'
import { Tensor, detach } from '..'
import { TensorValueIsNullError } from '../errors'

export class MatMul extends OpTrait {
  async compute(a: Tensor, b: Tensor) {
    const v1 = await a.raw
    const v2 = await b.raw
    if (v1 === null || v2 === null)
      throw new TensorValueIsNullError()
    return v1.matmul(v2)
  }

  /**
   * for Z = matmul(X, W),
   *
   * out_grad = dY / dZ
   *
   * grad X = matmul(out_grad, W.T)
   *
   * grad W = matmul(X.T, out_grad)
   *
   *  Parameter  |  size
   *  -----------|--------
   *  X          |  n x k
   *  W          |  k x m
   *  out_grad   |  n x m
   */
  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>, MaybePromise<Tensor>]) {
    const [ outGrad, lhs, rhs ] = await Promise.all([ grad, ...inputs ])
    return Promise.all([
      rhs.T.then(rT => detach(outGrad.matmul(rT))),
      lhs.T.then(lT => detach(lT.matmul(outGrad))),
    ])
  }
}

export async function matmul(a: MaybePromise<Tensor>, b: MaybePromise<Tensor>): Promise<Tensor> {
  const op = new MatMul()
  return Tensor.fromOp(op, a, b)
}
