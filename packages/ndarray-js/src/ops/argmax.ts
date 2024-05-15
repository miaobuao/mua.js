import type { NdArray, dtype } from '../ndarray'

export function argmax<T extends NdArray<dtype>>(lhs: T) {
  return Array.from(lhs.buffer.entries()).reduce((a, b) => a[1] > b[1] ? a : b)[0]
}
