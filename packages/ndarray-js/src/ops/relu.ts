import type { dtype } from '../ndarray'

import { NdArray } from '../ndarray'

export function relu<T extends NdArray<dtype>>(lhs: T): T {
  return new NdArray(
    lhs.buffer.map(d => Math.max(d, 0)),
    {
      shape: lhs.shape,
    },
  ) as T
}
