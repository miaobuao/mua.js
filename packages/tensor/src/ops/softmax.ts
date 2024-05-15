import type { MaybePromise } from '@mua/common'
import type { dtype } from 'ndarray-js'

import { isNil, range } from 'lodash-es'
import { NdArray, concat } from 'ndarray-js'

import { OpTrait } from './op-trait'
import { Tensor } from '..'
import { TensorValueIsNullError } from '../errors'

export class SoftmaxOps extends OpTrait {
  private ir: NdArray<dtype> | null = null

  constructor(
    readonly dim: number = -1,
  ) {
    super()
  }

  async compute(a: Tensor) {
    const arr = await a.raw
    if (arr === null)
      throw new TensorValueIsNullError()
    const res = await arr.softmax()
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
      ir = ir!.reshape([ 1, ir!.shape[0]! ])

    const res: NdArray[] = []
    const outGradRaw = await outGrad.raw

    for (let n = 0; n < gradSize[0]!; ++n) {
      const jacobian = new NdArray(range(gradSize[1]!).map(i => range(gradSize[1]!).map((j) => {
        const nj = ir!.slice(n, j).buffer[0]!
        if (i === j)
          return nj * (1 - nj)
        const ni = ir!.slice(n, i).buffer[0]!
        return -ni * nj
      })))

      const cell = outGradRaw?.matmul(jacobian)

      res.push(await cell!)
    }
    const resRaw = await concat(...res)

    return [
      new Tensor(
        await resRaw.reshape(originGradSize),
      ),
    ]
  }
}

export function softmax(a: MaybePromise<Tensor>, dim = -1) {
  const op = new SoftmaxOps(dim)
  return Tensor.fromOp(op, a)
}
