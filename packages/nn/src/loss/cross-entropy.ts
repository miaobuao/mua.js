import type { MaybePromise } from '@mua/common'

import { type Tensor, log, softmax } from '@mua/tensor'

import { nllloss } from './nllloss'
import { Module } from '../modules'

export class CrossEntropyLoss extends Module {
  async forward(x: MaybePromise<Tensor>, y: MaybePromise<Tensor>): Promise<Tensor> {
    let [ _x, _y ] = await Promise.all([ x, y ])
    _x = await log(softmax(_x))
    return nllloss(_x, _y)
  }
}
