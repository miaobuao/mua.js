import { type MaybePromise, getConv1dSize } from '@mua/common'
import { pipe } from 'fp-ts/lib/function'
import { inRange, range } from 'lodash-es'
import { NdArray, im2col } from 'ndarray'

import { TensorValueIsNullError } from '../../errors'
import { assert } from '../../helper'
import { Tensor } from '../../tensor'
import { OpTrait } from '../op-trait'

export interface Conv1dOpsParams {
  stride?: number
  padding?: number
  padValue?: number
}

class Conv1dOp extends OpTrait {
  readonly stride: number
  readonly padding: number
  readonly padValue: number
  readonly mapping: Map<number, number[]> = new Map()

  /** [k * cin, feat_out] */
  savedIm2cols: NdArray | null = null

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
    const wSize = weightMatrix.shape
    const convSize = getConv1dSize(inputSize, wSize, this.stride, this.padding)
    const [ kernelSize, cin, cout ] = wSize

    const inputRange: [number, number] = [ this.padding, this.padding + inputMatrix.shape[0]! ]
    /** [feat_in, cin] */
    // const paddedInput = await inputMatrix.padding(this.padValue, this.padding, this.padding)
    // this.savedIm2cols = pipe(
    range(convSize[0])
      . forEach((i) => {
        const st = i * this.stride
        const ed = st + kernelSize!
        if (ed >= this.padding && st < inputRange[1])
          this.mapping.set(i, range(st, ed))
      })
    //   transpose,
    // )
    this.savedIm2cols = im2col(
      inputMatrix,
      new Uint32Array([ kernelSize! ]),
      this.stride,
      new Uint32Array([ this.padding, this.padding ]),
      this.padValue,
    ).transpose()

    const W = pipe(
      /** [cout, k, cin] */
      await weightMatrix.permute(new Uint32Array([ 2, 0, 1 ])),
      /** [cout, k * cin] */
      v => v.reshape(new Int32Array([ cout!, kernelSize! * cin! ])),
    )

    const res = W.matmul(this.savedIm2cols!).transpose()

    // return .transpose()
    return res
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
    const inputGrad = NdArray.zeros(input.shape)

    // update inputGrad
    /** [k * cin, cout] */
    const wT = weight.reshape(new Int32Array([ kernelSize! * cin!, cout! ]))
    /** [N, K, cin] */
    const colGrad = pipe(
      wT.matmul(outGrad.transpose()).transpose(),
      (v) => {
        /** v: [N, k * cin] */
        return v.reshape(new Int32Array([ -1, kernelSize!, cin! ]))
      },
    )
    for (const [ idx, indices ] of this.mapping.entries()) {
      for (let index = 0; index < indices.length; ++index) {
        const i = indices[index]! - this.padding
        if (inRange(i, 0, input.shape[0]!)) {
          const grad = colGrad.slice(new Uint32Array([ idx, index ]))
          inputGrad.set(
            new Uint32Array([ i ]),
            inputGrad.slice(new Uint32Array([ i ])).add(grad),
          )
          // const grad = colGrad[idx]![index]!
          // inputGrad[i] = await add(
          //   inputGrad[i]!,
          //   grad,
          // ).then(d => d.value) as number[]
        }
      }
    }

    // update weightGrad
    /** [k * cin, feat_out] */
    const weightGrad = this.savedIm2cols?.matmul(outGrad).reshape(new Int32Array([ kernelSize!, cin!, cout! ]))

    return [
      new Tensor(inputGrad),
      new Tensor(weightGrad),
    ]
  }
}

export async function conv1d(input: MaybePromise<Tensor>, weight: MaybePromise<Tensor>, opts?: Conv1dOpsParams) {
  const op = new Conv1dOp(opts)
  return Tensor.fromOp(op, input, weight)
}
