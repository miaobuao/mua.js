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

class NLLLossOps extends OpTrait {
  constructor(readonly reduction: 'mean' | 'sum' = 'mean') {
    super()
    assert(reduction === 'mean' || reduction === 'sum', `NLLLoss: reduction should be mean or sum, not ${reduction}`)
  }

  async compute(pred: MaybePromise<Tensor>, target: MaybePromise<Tensor>) {
    const [ _pred, _target ] = await Promise.all([ pred, target ])
    const [ predRaw, targetRaw ] = await Promise.all([ _pred.raw, _target.raw ])
    if (isNil(predRaw) || isNil(targetRaw))
      throw new TensorValueIsNullError()
    const predSize = predRaw.shape as [number, number]
    const targetSize = targetRaw.shape as [number]
    if (predSize.length !== 2 || targetSize.length !== 1)
      throw new Error('NLLLoss: pred and target should be 2D and 1D tensors')
    if (predSize[0] !== targetSize[0])
      throw new Error('NLLLoss: pred and target should have the same batch size')
    const res: number[] = targetRaw.value.map((y, i) => predRaw.value[i][y])
    if (this.reduction === 'mean')
      return new NdArray([ -res.reduce((a, b) => a + b) / res.length ])
    else if (this.reduction === 'sum')
      return new NdArray([ -res.reduce((a, b) => a + b) ])
    else
      throw new Error(`NLLLoss: unknown reduction ${this.reduction}`)
  }

  /**
   * @param grad [ 1 ]
   * @param inputs input.0: [ batch_size, n_cls ]; input.1: [ batch_size ]
   */
  async gradient(grad: MaybePromise<Tensor>, ...inputs: [MaybePromise<Tensor>, MaybePromise<Tensor>]): Promise<[Tensor]> {
    const [ _grad, _pred, _target ] = await Promise.all([ grad, ...inputs ])
    const [ gradRaw, predRaw, targetRaw ] = await Promise.all([ _grad.raw, _pred.raw, _target.raw ])
    if (isNil(gradRaw) || isNil(predRaw) || isNil(targetRaw))
      throw new TensorValueIsNullError()
    const [ batchSize ] = predRaw.shape as [number, number]
    const inputGrad = await asyncValueNotNil(zeros(...predRaw.shape).raw)
    const [ outGrad ] = gradRaw.value
    const Y = targetRaw.value
    if (this.reduction === 'mean') {
      for (let i = 0; i < batchSize; ++i)
        inputGrad.value[i]![Y[i]] = -outGrad / batchSize
    }
    else if (this.reduction === 'sum') {
      for (let i = 0; i < batchSize; ++i)
        inputGrad[i][targetRaw.value[i]] = -outGrad
    }
    else {
      throw new Error(`NLLLoss: unknown reduction ${this.reduction}`)
    }
    return [ new Tensor(inputGrad) ]
  }
}

export function nllloss(pred: MaybePromise<Tensor>, target: MaybePromise<Tensor>, reduction: 'mean' | 'sum' = 'mean') {
  const op = new NLLLossOps(reduction)
  return Tensor.fromOp(op, pred, target)
}
