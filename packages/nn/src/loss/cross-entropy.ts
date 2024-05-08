import type { MaybePromise } from '@mua/common'

import { type Tensor, log, softmax } from '@mua/tensor'

import { nllloss } from './nllloss'
import { Module } from '../modules'

export class CrossEntropyLoss extends Module {
  async forward(x: MaybePromise<Tensor>, y: MaybePromise<Tensor>): Promise<Tensor> {
    x = softmax(x)
    x = log(x)
    return nllloss(x, y)
  }
}
