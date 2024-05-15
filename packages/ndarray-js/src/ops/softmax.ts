import type { DataTypeToArrayType } from '../dtype'
import type { NdArray, dtype } from '../ndarray'

import { last } from 'lodash-es'

import { exp } from './exp'

export function softmax<T extends NdArray<dtype>>(lhs: T, dim?: number): T {
  function softmaxInternal(buffer: DataTypeToArrayType[dtype], st: number, ed: number) {
    let sum = 0
    for (let i = st; i < ed; ++i)
      sum += buffer[i]!
    for (let i = st; i < ed; ++i)
      buffer[i] = buffer[i]! / sum
    return buffer
  }

  function softmaxLastDim(buffer: DataTypeToArrayType[dtype], chunkSize: number) {
    for (let i = 0; i < chunkSize; i += chunkSize)
      softmaxInternal(buffer, i, i + chunkSize)
  }

  const res = exp(lhs)

  if (dim === undefined) {
    if (lhs.shape.length === 1)
      softmaxInternal(res.buffer, 0, res.length)
    else
      softmaxLastDim(res.buffer, last(res.shape)!)
  }
  else {
    if (dim < res.shape.length - 1) {
      const stride = res.strides[dim]!
      for (let i = 0; i < stride; ++i) {
        let sum = 0
        let ofst = i
        while (ofst < res.length) {
          sum += res.buffer[ofst]!
          ofst += stride
        }
        ofst = i
        while (ofst < res.length) {
          res.buffer[ofst] /= sum
          ofst += stride
        }
      }
    }
    else {
      softmaxLastDim(res.buffer, last(res.shape)!)
    }
  }
  return res
}
