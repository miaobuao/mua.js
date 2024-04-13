import { getConv1dSize } from '@mua/common'
import { range } from 'lodash-es'
import { type Matrix, add, matrix, multiply } from 'mathjs'

/**
 *
 * @param input  [n, cin]
 * @param weight [k, cin, cout]
 * @returns [m, cout]
 *
 * m = (n - k + 2 * padding) / stride + 1
 */
export async function conv1d(
  input: Matrix,
  weight: Matrix,
  {
    stride,
    padding,
    padValue,
  }: {
    stride?: number
    padding?: number
    padValue?: number
  } = { stride: 1, padding: 0, padValue: 0 },
) {
  padding ??= 0
  if (padding < 0)
    throw new Error(`conv1d: padding must be greater than 0`)
  padValue ??= 0

  stride ??= 1
  if (stride <= 0)
    throw new Error(`conv1d: stride must be greater than 0`)

  const inputSize = input.size()
  const wSize = weight.size()
  const convSize = getConv1dSize(inputSize, wSize, stride, padding)
  const kernelSize = wSize[0]!

  let paddedInput = input.toArray() as number[][]
  if (padding > 0) {
    const pad = range(padding).map(() => Array(inputSize[1]!).fill(padValue))
    paddedInput = [
      pad,
      paddedInput,
      pad,
    ].flat()
  }

  const weightArr = weight.toArray() as unknown as number[][][]
  const res: number[][] = []
  for (let i = 0; i < convSize[0]; i++) {
    res.push(
      paddedInput
        .slice(i * stride, i * stride + kernelSize)
        .map((embed, idx) => {
          const res = multiply([ embed ], weightArr[idx]!)
          return res.flat()
        })
        .reduce((a, b) => add(a, b))
        .flat(),
    )
  }

  return matrix(res)
}
