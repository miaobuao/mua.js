import type { MaybePromise } from '@mua/common'

import { isNil } from 'lodash-es'
import { type NdArray, concat } from 'ndarray'

import { OpTrait } from './op-trait'
import { Tensor } from '..'
import { TensorValueIsNullError } from '../errors'

export class SoftmaxOps extends OpTrait {
  private ir: NdArray | null = null

  async compute(a: Tensor) {
    const arr = await a.raw
    if (arr === null)
      throw new TensorValueIsNullError()
    const res = arr.softmax()
    this.ir = res
    return res
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]) {
    let [ outGrad, input ] = await Promise.all([ grad, ...inputs ])
    const originGradSize = await outGrad.shape
    if (originGradSize.length === 1)
      outGrad = await outGrad.reshape([ 1, originGradSize[0]! ])
    const gradSize = await outGrad.shape
    let ir = this.ir
    if (isNil(ir))
      ir = await this.compute(input)
    if (ir!.shape.length === 1)
      ir = ir!.reshape(new Int32Array([ 1, ir!.shape[0]! ]))

    let res: NdArray | undefined
    for (let n = 0; n < gradSize[0]!; ++n) {
      const jacobian = ir.dot(ir.mulScalar(-1).addScalar(1))
      const cell = await outGrad.raw.then(raw => raw!.matmul(jacobian))
      if (res === undefined)
        res = cell
      else
        res = concat(res, cell)
    }

    return [
      new Tensor(
        await res!.reshape(new Int32Array(originGradSize)),
      ),
    ]
  }
}

export function softmax(a: MaybePromise<Tensor>) {
  const op = new SoftmaxOps()
  return Tensor.fromOp(op, a)
}
