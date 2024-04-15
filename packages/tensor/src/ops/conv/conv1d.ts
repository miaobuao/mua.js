import type { MaybePromise } from '@mua/common'
import type { NdArray } from 'async-math'

import { getConv1dSize } from '@mua/common'
import { Reshape, add, matmul, reshape, transpose, zeros } from 'async-math'
import { pipe } from 'fp-ts/lib/function'
import { inRange, range } from 'lodash-es'

import { TensorValueIsNullError } from '../../errors'
import { assert } from '../../helper'
import { Tensor } from '../../tensor'
import { OpTrait } from '../op-trait'

export interface Conv1dOpsParams {
  stride?: number
  padding?: number
  padValue?: number
}

class Conv1d extends OpTrait {
  readonly stride: number
  readonly padding: number
  readonly padValue: number
  readonly mapping: Map<number, number[]> = new Map()

  /** [k * cin, feat_out] */
  savedIm2cols: number[][] = []

  constructor(
    {
      stride,
      padding,
      padValue,
    }: Conv1dOpsParams = {},
  ) {
    super()
    this.padding = padding ?? 0

    assert(this.padding >= 0, `conv1d: padding must be greater or equal than 0, got ${this.padding}`)

    this.stride = stride ?? 1
    assert(this.stride > 0, `conv1d: stride must be greater than 0, got ${this.stride}`)

    this.padValue = padValue ?? 0
  }

  /**
   *
   * @param input [feat_in, cin]
   * @param weight [k, cin, cout]
   * @returns [feat_out, cout]
   */
  async compute(input: MaybePromise<Tensor>, weight: MaybePromise<Tensor>) {
    this.mapping.clear()

    const [ _input, _weight ] = await Promise.all([ input, weight ])
    const [ inputMatrix, weightMatrix ] = await Promise.all([ _input.raw, _weight.raw ])
    if (!inputMatrix || !weightMatrix)
      throw new TensorValueIsNullError()

    const inputSize = inputMatrix.shape
    const wSize = weightMatrix.shape as [number, number, number]
    const convSize = getConv1dSize(inputSize, wSize, this.stride, this.padding)
    const [ kernelSize, cin, cout ] = wSize

    let paddedInput = inputMatrix.toArray() as number[][]
    const inputRange: [number, number] = [ this.padding, this.padding + paddedInput.length ]
    if (this.padding > 0) {
      const pad = range(this.padding).map(() => Array(inputSize[1]!).fill(this.padValue))
      paddedInput = [
        pad,
        paddedInput,
        pad,
      ].flat()
    }

    this.savedIm2cols = pipe(
      range(convSize[0])
        . map((i) => {
          const st = i * this.stride
          const ed = st + kernelSize
          if (ed >= this.padding && st < inputRange[1])
            this.mapping.set(i, range(st, ed))
          return paddedInput.slice(st, ed).flat()
        }),
      transpose,
    )

    const W = pipe(
      /** [cout, k, cin] */
      await weightMatrix.permute([ 2, 0, 1 ]),
      /** [cout, k * cin] */
      Reshape([ cout, kernelSize * cin ]),
    ) as NdArray
    return (await matmul(W, this.savedIm2cols)).T
  }

  /**
   *
   * @param grad [out_feat, cout]
   * @param inputs input: [in_feat, cin], weight: [k, cin, cout]
   * @returns  [in_feat, cin], [k, cin, cout]
   */
  async gradient(
    grad: MaybePromise<Tensor>,
    ...inputs: [MaybePromise<Tensor>, MaybePromise<Tensor>]
  ): Promise<[Tensor, Tensor]> {
    const [ outGrad, input, weight ] = await Promise.all(
      [
        grad,
        ...inputs,
      ].map(d => Promise.resolve(d).then(d => d.raw)),
    )
    if (!outGrad || !input || !weight)
      throw new TensorValueIsNullError()

    const [ kernelSize, cin, cout ] = weight.shape
    /** [feat_in, cin] */
    const inputGrad = zeros(input.shape) as number[][]

    // update inputGrad
    /** [k * cin, cout] */
    const wT = reshape(weight.value, [ kernelSize! * cin!, cout! ])
    /** [N, K, cin] */
    const colGrad = pipe(
      await matmul(wT, outGrad.T).then(d => d.T),
      (v) => {
        /** [N, k * cin] */
        const grads = v.value as number[][]
        return grads.map(d => reshape(d, [ kernelSize!, cin! ]) as unknown as number[][])
      },
    )
    for (const [ idx, indices ] of this.mapping.entries()) {
      for (let index = 0; index < indices.length; ++index) {
        const i = indices[index]! - this.padding
        if (inRange(i, 0, input.shape[0]!)) {
          const grad = colGrad[idx]![index]!
          inputGrad[i] = await add(
            inputGrad[i]!,
            grad,
          ).then(d => d.value) as number[]
        }
      }
    }

    // update weightGrad
    /** [k * cin, feat_out] */
    const weightGrad = pipe(
      await matmul(this.savedIm2cols, outGrad.value),
      Reshape([ kernelSize!, cin!, cout! ]),
    )

    return [
      new Tensor(inputGrad),
      new Tensor(weightGrad),
    ]
  }
}

export async function conv1d(input: MaybePromise<Tensor>, weight: MaybePromise<Tensor>, opts?: Conv1dOpsParams) {
  const op = new Conv1d(opts)
  return Tensor.fromOp(op, input, weight)
}
