import type { dtype } from '../ndarray'

import { NdArray } from '../ndarray'

export function sub<T extends NdArray<dtype>>(lhs: T, rhs: T | number): T {
  if (typeof rhs === 'number') {
    return new NdArray(
      lhs.buffer.map(d => d - rhs),
      {
        shape: lhs.shape,
      },
    ) as T
  }
  return new NdArray(
    lhs.buffer.map((d, i) => d - rhs.buffer[i]!),
    {
      shape: lhs.shape,
    },
  ) as T
}
