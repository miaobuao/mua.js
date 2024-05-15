import type { dtype } from '../ndarray'

import { NdArray } from '../ndarray'

export function log<T extends NdArray<dtype>>(lhs: T, base: number = Math.E): T {
  const LN = Math.log(base)
  return new NdArray(
    lhs.buffer.map(d => Math.log(d) / LN),
    {
      shape: lhs.shape,
    },
  ) as T
}
