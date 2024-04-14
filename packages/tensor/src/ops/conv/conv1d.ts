import type { ConvOpsParams } from './interface'
import type { MaybePromise } from '../../helper'
import type { NdArrayNumberCell } from 'async-math'

import { getConv1dSize } from '@mua/common'
import { NdArray, add, matmul, transpose, zeros } from 'async-math'
import { filter, map } from 'fp-ts/lib/Array'
import { pipe } from 'fp-ts/lib/function'
import { inRange, range } from 'lodash-es'

import { TensorValueIsNullError } from '../../errors'
import { Tensor } from '../../tensor'
import { OpTrait } from '../op-trait'

class Conv1d extends OpTrait {
  readonly stride: number
  readonly padding: number
  readonly padValue: number
  readonly mapping: Map<number, number[]> = new Map()

  /** [k, feat_out, cout] */
  savedIm2cols: NdArray[] = []

  constructor(
    {
      stride,
      padding,
      padValue,
    }: ConvOpsParams = {},
  ) {
    super()
    this.padding = padding ?? 0

    if (this.padding < 0)
      throw new Error(`conv1d: padding must be greater than 0`)

    this.stride = stride ?? 1
    if (this.stride <= 0)
      throw new Error(`conv1d: stride must be greater than 0`)

    this.padValue = padValue ?? 0
  }

  /**
   *
   * @param input [n, cin]
   * @param weight [k, cin, cout]
   * @returns [m, cout]
   *
   * m = (n + padding * 2 - k)
   */
  async compute(input: MaybePromise<Tensor>, weight: MaybePromise<Tensor>) {
    this.mapping.clear()

    const [ _input, _weight ] = await Promise.all([ input, weight ])
    const [ inputMatrix, weightMatrix ] = await Promise.all([ _input.raw, _weight.raw ])
    if (!inputMatrix || !weightMatrix)
      throw new TensorValueIsNullError()

    const inputSize = inputMatrix.shape
    const wSize = weightMatrix.shape
    const convSize = getConv1dSize(inputSize, wSize, this.stride, this.padding)
    const kernelSize = wSize[0]!

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

    const weightArr = weightMatrix.toArray() as number[][][]

    this.savedIm2cols = range(convSize[0]).map((i) => {
      const st = i * this.stride
      const ed = st + kernelSize
      if (ed >= this.padding && st < inputRange[1]) {
        const indices = pipe(
          range(st, ed),
          filter(v => inRange(v, ...inputRange)),
          map(d => d - this.padding),
        )
        this.mapping.set(i, indices)
      }
      return paddedInput.slice(st, ed)
    }).reduce(
      (a, b) => range(kernelSize).map(i => [ ...a[i]!, b[i]! ]),
      range(kernelSize).map(() => []) as number[][][],
    ).map(d => new NdArray(d))

    const feats = await Promise.all(this.savedIm2cols.map(
      (d, i) => matmul(d, weightArr[i]!),
    ))
    let res = feats[0]!
    for (let i = 1; i < feats.length; i++)
      res = await add(res, feats[i]!)

    return await res
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

    const inputGrad = zeros(input.shape) as number[][]
    const weightGrad = zeros(weight.shape) as unknown as NdArrayNumberCell[]

    // update inputGrad
    for (const [ idx, indices ] of this.mapping.entries()) {
      const effectRow: number[] = outGrad.value[idx] // size: [ cout ]
      /** size: [k, cin] */
      const grads = await Promise.all(weight.value.map((k) => {
        const kT: number[][] = transpose(k) // kT: size: [cout, cin]
        return matmul([ effectRow ], kT)
      }))
      indices.forEach(async (i) => {
        inputGrad[i] = await add(inputGrad[i]!, grads[0]!).then(d => d.value) as number[]
      })
    }

    // update weightGrad
    await Promise.all(this.savedIm2cols.map(async ({ value: col }, i) => {
      /**
       * col.shape: [feat_out, cin]
       * outGrad.shape: [feat_out, cout]
       * kernel.shape: [cin, cout],
       *
       * grad = col.T * outGrad
       */
      const grad = await matmul(transpose(col as number[][]) as number[][], outGrad.value)
      weightGrad[i] = await add(weightGrad[i]!, grad).then(d => d.value)
    }))

    return [
      new Tensor(inputGrad),
      new Tensor(weightGrad),
    ]
  }
}

export async function conv1d(input: MaybePromise<Tensor>, weight: MaybePromise<Tensor>, opts?: ConvOpsParams) {
  const op = new Conv1d(opts)
  return Tensor.fromOp(op, input, weight)
}
