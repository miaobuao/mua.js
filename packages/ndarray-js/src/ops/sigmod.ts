import type { dtype } from '../ndarray'

import { NdArray } from '../ndarray'

export function sigmod<T extends NdArray<dtype>>(lhs: T): T {
  return new NdArray(
    lhs.buffer.map(d => 1 / (1 + Math.exp(-d))),
    {
      shape: lhs.shape,
    },
  ) as T
}
