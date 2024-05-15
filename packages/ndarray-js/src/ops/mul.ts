import type { dtype } from '../ndarray'

import { NdArray } from '../ndarray'

export function dot<T extends NdArray<dtype>>(lhs: T, rhs: T): T {
  return new NdArray(
    lhs.buffer.map((d, i) => d * rhs.buffer[i]!),
    {
      shape: lhs.shape,
    },
  ) as T
}

export function mul<T extends NdArray<dtype>>(lhs: T, rhs: number | T): T {
  if (typeof rhs === 'number') {
    return new NdArray(
      lhs.buffer.map(d => d * rhs),
      {
        shape: lhs.shape,
      },
    ) as T
  }
  return dot(lhs, rhs)
}
