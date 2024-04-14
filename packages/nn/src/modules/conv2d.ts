import type { Conv1dOpsParams, MaybePromise, Tensor } from '@mua/tensor'

import { conv2d, randn } from '@mua/tensor'

import { Module } from './module'

export class Conv2d extends Module {
  readonly stride: number
  readonly padding: number
  readonly padValue: number
  readonly weight: Tensor

  constructor(inChannels: number, outChannels: number, kernelSize: number, opts?: Conv1dOpsParams) {
    super()
    this.stride = opts?.stride ?? 1
    this.padding = opts?.padding ?? 0
    this.padValue = opts?.padValue ?? 0
    this.weight = randn([ kernelSize, kernelSize, inChannels, outChannels ])
  }

  async forward(x: MaybePromise<Tensor>): Promise<Tensor> {
    const input = await x
    return conv2d(
      input,
      this.weight,
      {
        stride: this.stride,
        padding: this.padding,
        padValue: this.padValue,
      },
    )
  }
}
