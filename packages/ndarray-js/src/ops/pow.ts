import type { dtype } from '../ndarray'

import { NdArray } from '../ndarray'

export function pow<T extends NdArray<dtype>>(lhs: T, exponent: number): T {
  return new NdArray(
    lhs.buffer.map(d => d ** exponent),
    {
      shape: lhs.shape,
    },
  ) as T
}
