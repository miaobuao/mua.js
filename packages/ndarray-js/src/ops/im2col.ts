import type { DataTypeToArrayType } from '../dtype'
import type { dtype } from '../ndarray'

import { last } from 'lodash-es'

import { concatTypedArray, ndIdxToOffset } from '../lib'
import { NdArray } from '../ndarray'

export function im2col<T extends NdArray<dtype>>(lhs: T, kernelSize: number[], stride?: number): T {
  if (lhs.shape.length === 2)
    lhs = lhs.reshape([ 1, ...lhs.shape ])
  else if (lhs.shape.length > 3 || lhs.shape.length < 2)
    throw new Error('im2col only support 2D or 3D input')

  if (kernelSize.length === 1)
    kernelSize = [ 1, ...kernelSize ]
  else if (kernelSize.length > 3 || kernelSize.length < 2)
    throw new Error('im2col only support 2D or 3D kernel')

  stride ||= 1

  // TODO: support more than 3D
  const chunkSize = kernelSize[0]! * kernelSize[1]! * last(lhs.shape)!
  let cnt = 0
  const res: DataTypeToArrayType[dtype][] = []
  for (let i = 0; i < lhs.shape[0]! / stride; ++i) {
    const r = i * stride
    if (r + kernelSize[0]! > lhs.shape[0]!)
      continue
    for (let j = 0; j < lhs.shape[1]! / stride; ++j) {
      const c = j * stride
      if (c + kernelSize[1]! > lhs.shape[1]!)
        continue
      const chunk: DataTypeToArrayType[dtype][] = []
      for (let k = 0; k < kernelSize[0]!; ++k) {
        const ed = ndIdxToOffset([ r + k, c + kernelSize[1]!, 0 ], lhs.strides)
        if (ed > lhs.length)
          break
        const st = ndIdxToOffset([ r + k, c, 0 ], lhs.strides)
        const slice = lhs.buffer.slice(st, ed)
        chunk.push(slice)
      }
      const buf = concatTypedArray(chunk)
      if (buf.length !== chunkSize)
        continue
      res.push(buf)
      cnt += 1
    }
  }
  return new NdArray(
    concatTypedArray(res),
    {
      shape: [ cnt, chunkSize ],
    },
  ) as T
}
