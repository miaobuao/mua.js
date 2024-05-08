import type { MaybePromise } from '@mua/common'

import { isNil, range } from 'lodash-es'
import { type NdArray, concat } from 'ndarray'

import { OpTrait } from './op-trait'
import { Tensor, toNdArray } from '..'
import { TensorValueIsNullError } from '../errors'

export class SoftmaxOps extends OpTrait {
  private ir: NdArray | null = null

  constructor(
    readonly dim: number = -1,
  ) {
    super()
  }

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
    if (ir.shape.length === 1)
      ir = ir.reshape(new Int32Array([ 1, ir!.shape[0]! ]))

    let res: NdArray | undefined
    const outGradRaw = await outGrad.raw

    for (let n = 0; n < gradSize[0]!; ++n) {
      const jacobian = toNdArray(range(gradSize[1]!).map(i => range(gradSize[1]!).map((j) => {
        const nj = ir.slice(new Uint32Array([ n, j ])).buffer[0]!
        if (i === j)
          return nj * (1 - nj)
        const ni = ir.slice(new Uint32Array([ n, i ])).buffer[0]!
        return -ni * nj
      })))

      const cell = outGradRaw?.matmul(jacobian)

      if (res === undefined)
        res = cell
      else
        res = concat(res, cell!)
    }

    return [
      new Tensor(
        await res!.reshape(new Int32Array(originGradSize)),
      ),
    ]
  }
}

export function softmax(a: MaybePromise<Tensor>, dim = -1) {
  const op = new SoftmaxOps(dim)
  return Tensor.fromOp(op, a)
}
