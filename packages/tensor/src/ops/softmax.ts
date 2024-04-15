import type { MaybePromise } from '@mua/common'

import { NdArray, softmax as _sm, matmul } from 'async-math'
import { isNil, range } from 'lodash-es'

import { OpTrait } from './op-trait'
import { Tensor } from '..'
import { TensorValueIsNullError } from '../errors'

export class SoftmaxOps extends OpTrait {
  private ir: NdArray | null = null

  async compute(a: Tensor) {
    const arr = await a.raw
    if (arr === null)
      throw new TensorValueIsNullError()
    const res = new NdArray(await _sm(arr.value) as number[])
    this.ir = res
    return res
  }

  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>]) {
    let [ outGrad, input ] = await Promise.all([ grad, ...inputs ])
    const originGradSize = await outGrad.shape
    if (originGradSize.length === 1)
      outGrad = await outGrad.reshape([ 1, originGradSize[0]! ])
    const gradSize = await outGrad.shape as [number, number]
    let ir = this.ir
    if (isNil(ir))
      ir = await this.compute(input)
    if (ir.shape.length === 1)
      ir = ir.reshape([ 1, ir.shape[0]! ])

    let res = new NdArray<number>([])
    for (let n = 0; n < gradSize[0]; ++n) {
      const jacobian = range(gradSize[1]).map(i => range(gradSize[1]).map((j) => {
        if (i === j)
          return ir.value[n]![j]! * (1 - ir.value[n]![j]!)
        return -ir.value[n]![i] * ir.value[n]![j]
      }))
      const cell = await matmul(
        outGrad.raw.then(raw => raw?.value[n]),
        jacobian,
      )
      res = res.concat(cell.value)
    }

    return [
      new Tensor(
        await res.reshape(originGradSize),
      ),
    ]
  }
}

export function softmax(a: MaybePromise<Tensor>) {
  const op = new SoftmaxOps()
  return Tensor.fromOp(op, a)
}
