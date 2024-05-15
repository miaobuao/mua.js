import type { dtype } from '../ndarray'

import { NdArray } from '../ndarray'

export function exp<T extends NdArray<dtype>>(lhs: T): T {
  return new NdArray(
    lhs.buffer.map(d => Math.exp(d)),
    {
      shape: lhs.shape,
    },
  ) as T
}
