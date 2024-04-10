import { matmul as _matmul } from 'async-math'

import { OpTrait } from '.'
import { Tensor, detach } from '..'
import { TensorValueIsNullError, TensorValueTypeError } from '../errors'

export class MatMul extends OpTrait {
  async compute(a: Tensor, b: Tensor) {
    const v1 = await a.raw
    const v2 = await b.raw
    if (v1 === null || v2 === null)
      throw new TensorValueIsNullError()
    if (typeof v1 === 'number' || typeof v2 === 'number')
      throw new TensorValueTypeError()
    return _matmul(v1, v2)
  }

  /**
   * Compute the gradient of a matrix multiplication
   *
   * for Z = matmul(X, W),
   *
   * out_grad = dY / dZ
   *
   * grad X = matmul(out_grad, W.T)
   *
   * grad W = matmul(X.T, out_grad)
   */
  async gradient(grad: Tensor, ...inputs: [Tensor, Tensor]) {
    const [ lhs, rhs ] = inputs
    return Promise.all([
      rhs.T.then(rT => detach(grad.matmul(rT))),
      lhs.T.then(lT => detach(lT.matmul(grad))),
    ])
  }
}

export function matmul(a: Tensor, b: Tensor): Promise<Tensor> {
  const op = new MatMul()
  return Tensor.fromOp(op, a, b)
}
