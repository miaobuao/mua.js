import type { dtype } from '../ndarray'

import { NdArray } from '../ndarray'

export function tanh<T extends NdArray<dtype>>(lhs: T): T {
  return new NdArray(
    lhs.buffer.map(d => Math.tanh(d)),
    {
      shape: lhs.shape,
    },
  ) as T
}
