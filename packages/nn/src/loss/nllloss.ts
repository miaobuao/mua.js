import { type MaybePromise, asyncValueNotNil } from '@mua/common'
import { NdArray, OpTrait, Tensor, TensorValueIsNullError, assert, reshape, zeros } from '@mua/tensor'
import { isNil } from 'lodash-es'

import { Module } from '../modules'

export class NLLLoss extends Module {
  constructor(
    public reduction: 'mean' | 'sum' = 'mean',
  ) { super() }

  async forward(pred: MaybePromise<Tensor>, target: MaybePromise<Tensor>): Promise<Tensor> {
    let [ _pred, _target ] = await Promise.all([ pred, target ])
    const predSize = await _pred.shape
    if (predSize.length === 1)
      _pred = await reshape(_pred, [ 1, predSize[0]! ])
    return nllloss(pred, target, this.reduction)
  }
}

class NLLLossOp extends OpTrait {
  constructor(readonly reduction: 'mean' | 'sum' = 'mean') {
    super()
    assert(reduction === 'mean' || reduction === 'sum', `NLLLoss: reduction should be mean or sum, not ${reduction}`)
  }

  async compute(pred: MaybePromise<Tensor>, target: MaybePromise<Tensor>) {
    const [ _pred, _target ] = await Promise.all([ pred, target ])
    const [ predRaw, targetRaw ] = await Promise.all([ _pred.raw, _target.raw ])
    if (isNil(predRaw) || isNil(targetRaw))
      throw new TensorValueIsNullError()
    const predSize = predRaw.shape
    const targetSize = targetRaw.shape
    if (predSize.length !== 2 || targetSize.length !== 1)
      throw new Error('NLLLoss: pred and target should be 2D and 1D tensors')
    if (predSize[0] !== targetSize[0])
      throw new Error('NLLLoss: pred and target should have the same batch size')

    const res = targetRaw.buffer.map((label, index) => predRaw.slice(index).buffer[label]!)

    if (this.reduction === 'mean')
      return new NdArray(new Float32Array([ -res.reduce((a, b) => a + b) / res.length ]))
    else if (this.reduction === 'sum')
      return new NdArray(new Float32Array([ -res.reduce((a, b) => a + b) ]))
    else
      throw new Error(`NLLLoss: unknown reduction ${this.reduction}`)
  }

  /**
   * @param grad [ 1 ]
   * @param inputs input.0: [ batch_size, n_cls ]; input.1: [ batch_size ]
   */
  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>, MaybePromise<Tensor>]): Promise<[Tensor]> {
    const [ _grad, _pred, _target ] = await Promise.all([ grad, ...inputs ])
    const [ outGrad, predRaw, Y ] = await Promise.all([ _grad.raw, _pred.raw, _target.raw ])

    if (isNil(outGrad) || isNil(predRaw) || isNil(Y))
      throw new TensorValueIsNullError()

    const [ batchSize ] = predRaw.shape
    const inputGrad = await asyncValueNotNil(zeros(predRaw.shape).raw)

    if (this.reduction === 'mean') {
      Y.buffer.forEach(async (label, index) => {
        inputGrad.set(
          [ index, label ],
          await outGrad.mul(-1 / batchSize!),
        )
      })
    }
    else if (this.reduction === 'sum') {
      Y.buffer.forEach(async (label, index) => {
        inputGrad.set(
          [ index, label ],
          await outGrad.mul(-1),
        )
      })
    }
    else {
      throw new Error(`NLLLoss: unknown reduction ${this.reduction}`)
    }
    return [ new Tensor(inputGrad) ]
  }
}

export function nllloss(pred: MaybePromise<Tensor>, target: MaybePromise<Tensor>, reduction: 'mean' | 'sum' = 'mean') {
  const op = new NLLLossOp(reduction)
  return Tensor.fromOp(op, pred, target)
}
