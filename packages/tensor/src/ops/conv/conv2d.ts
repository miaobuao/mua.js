import type { MaybePromise } from '@mua/common'

import { getConv2dSize } from '@mua/common'
import { pipe } from 'fp-ts/lib/function'
import { inRange, isNil, range } from 'lodash-es'
import { NdArray, im2col } from 'ndarray'

import { TensorValueIsNullError } from '../../errors'
import { assert } from '../../helper'
import { Tensor } from '../../tensor'
import { OpTrait } from '../op-trait'

interface Conv2dOpsParams {
  stride?: number
  padding?: number
  padValue?: number
}

class Conv2d extends OpTrait {
  readonly stride: number
  readonly padding: number
  readonly padValue: number
  /**
    map from origin to convolutional position
   */
  private mapping: Map<number, Map<number, [number, number][]>> = new Map()

  /**
   * shape: [h_out * w_out, kh * kw * cin]
   */
  savedIm2cols: NdArray | null = null

  constructor(
    {
      stride,
      padding,
      padValue,
    }: Conv2dOpsParams = {},
  ) {
    super()

    this.padding = padding ?? 0
    assert(this.padding >= 0, `conv2d: padding must be greater or equal than 0, got ${this.padding}`)

    this.stride = stride ?? 1
    assert(this.stride > 0, `conv2d: stride must be greater than 0, got ${this.stride}`)

    this.padValue = padValue ?? 0
  }

  /**
   * @param input [h, w, cin]
   * @param weight [ kh, kw, cin, cout]
   * @returns [h_out, w_out, cout]
   */
  async compute(input: MaybePromise<Tensor>, weight: MaybePromise<Tensor>) {
    this.mapping.clear()
    const [ inputMatrix, weightMatrix ] = await Promise.all(
      pipe(
        await Promise.all([ input, weight ]),
        ([ x, w ]) => [ x.raw, w.raw ],
      ),
    )
    if (!inputMatrix || !weightMatrix)
      throw new TensorValueIsNullError()
    const [ hin, win ] = inputMatrix.shape
    const [ kh, kw, cin, cout ] = weightMatrix.shape
    /** paddedInput.size == [hout, wout, cin] */
    // let paddedInput: number[][][] = inputMatrix
    // if (this.padding > 0) {
    //   paddedInput = padLr2d(paddedInput, this.padding, this.padValue)
    //   paddedInput = padUd2d(paddedInput, this.padding, this.padValue)
    // }

    const [ hout, wout ] = getConv2dSize(inputMatrix.shape, weightMatrix.shape, this.stride, this.padding)

    // this.savedIm2cols =

    range(hout).forEach((h) => {
      range(wout).forEach((w) => {
        const startH = h * this.stride
        const startW = w * this.stride
        const endH = startH + kh!
        const endW = startW + kw!
        for (let i = startH; i < endH; i++) {
          /** origin position */
          const ii = i - this.padding
          const map = this.mapping.get(ii) ?? new Map()
          this.mapping.set(ii, map)
          for (let j = startW; j < endW; j++) {
            /** origin position */
            const jj = j - this.padding
            if (inRange(ii, 0, hin) && inRange(jj, 0, win)) {
              const record = map.get(jj) ?? []
              map.set(jj, record)
              record.push([ h, w ])
            }
          }
        }
        // const part = pipe(
        //   /** [kh, kw, cin] */
        //   paddedInput.slice(startH, endH).map(row => row.slice(startW, endW)),
        //   /** [kh * kw * cin] */
        //   Reshape([ kh * kw * cin ]),
        // ) as unknown as number[]
        // return part
      })
    })
    this.savedIm2cols = im2col(inputMatrix, new Uint32Array([ kh!, kw! ]), this.stride, new Uint32Array([ this.padding, this.padding ]), this.padValue)
    const flatKernel = weightMatrix.reshape(new Int32Array([ kh! * kw! * cin!, cout! ]))
    const res = this.savedIm2cols.matmul(flatKernel)
    return res.reshape(new Int32Array([ hout!, wout!, cout! ]))
    // const res = await matmul(this.savedIm2cols, flatKernel).then(NdArray.toValue)
    // return new NdArray(reshape(res as number[], [ hout, wout, cout ]) as NdArrayNumberCell[])
  }

  /**
   *
   * @param grad [h_out, w_out, cout]
   * @param inputs input: [h, w, cin], weight: [kh, kw, cin, cout]
   * @returns  [h, w, cin], [kh, kw, cin, cout]
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

    const [ _, __, cin ] = input.shape
    const [ kh, kw ] = weight.shape
    const [ hout, wout, cout ] = outGrad.shape
    /** [h, w, cin] */
    const inputGrad = NdArray.zeros(input.shape)

    // update inputGrad
    /** [cout, kh * kw * cin] */
    const deconvKernel = weight.reshape(new Int32Array([ kh! * kw! * cin!, cout! ])).transpose()
    for (let h = 0; h < hout!; ++h) {
      const indicesMap = this.mapping[h]
      if (isNil(indicesMap))
        continue
      for (let w = 0; w < wout!; ++w) {
        const indices = indicesMap[w]
        if (isNil(indices))
          continue
        /** [cout] */
        const grad = outGrad.slice(new Uint32Array([ h, w ]))
        /** [1, kh * kw * cin] */
        const deconved = grad.matmul(deconvKernel)
        /** [kh, kw, cin] */
        const grads = deconved.reshape(new Int32Array([ kh!, kw!, cin! ]))
        const offsetY = h * this.stride
        const offsetX = w * this.stride
        for (const [ i, j ] of indices) {
          const lG = inputGrad.slice(new Uint32Array([ i - this.padding, j - this.padding ]))
          inputGrad.set(
            new Uint32Array([ i - this.padding, j - this.padding ]),
            lG.add(grads.slice(new Uint32Array([ i - offsetY, j - offsetX ]))),
          )
          // inputGrad[i - this.padding]![j - this.padding] = await add(lG, grads.value[i - offsetY]![j - offsetX]).then(NdArray.toValue)
        }
      }
    }

    // update weightGrad
    /** [kh * kw * cin, hout * wout] */
    const transposedCol = this.savedIm2cols?.transpose()
    /** [hout * wout, cout] */
    const flatGrad = outGrad.flatten().reshape(new Int32Array([ hout! * wout!, cout! ]))

    /** [kh, kw, cin, cout] */
    const weightGrad = transposedCol?.matmul(flatGrad).reshape(new Int32Array([ kh!, kw!, cin!, cout! ]))

    return [
      new Tensor(inputGrad),
      new Tensor(weightGrad),
    ]
  }
}

/** pad 2d left and right */
// export function padLr2d(input: number[][][], padding: number, value: number) {
//   if (padding === 0)
//     return input
//   const shape = size(input as any) as [number, number, number]
//   const pad: number[][] = Array(padding).fill(Array(shape[2]).fill(value))
//   return input.map(d => [ ...pad, ...d, ...pad ])
// }

/** pad 2d up and down */
// export function padUd2d(input: number[][][], padding: number, value: number) {
//   if (padding === 0)
//     return input
//   const [ _, w, c ] = size(input as any) as [number, number, number]
//   const pad: number[][][] = Array(padding).fill(Array(w).fill(Array(c).fill(value)))
//   return [
//     ...pad,
//     ...input,
//     ...pad,
//   ]
// }

export async function conv2d(input: MaybePromise<Tensor>, weight: MaybePromise<Tensor>, opts?: Conv2dOpsParams) {
  const op = new Conv2d(opts)
  return Tensor.fromOp(op, input, weight)
}
