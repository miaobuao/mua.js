import type { dtype } from '../ndarray'

import { arrayInit } from '../lib'
import { NdArray } from '../ndarray'

export function concat<T extends NdArray<dtype>>(...args: T[]): T {
  const shape = [ ...args[0]!.shape ]
  for (let i = 1; i < args.length; ++i)
    shape[0] += args[i]!.shape[0]!
  const type = args[0]!.dtype
  const buf = arrayInit(shape.reduce((a, b) => a * b), type)
  let offset = 0
  for (const b of args) {
    buf.set(b.buffer, offset)
    offset += b.buffer.length
  }
  return new NdArray(
    buf,
    {
      shape,
      dtype: type,
    },
  ) as T
}
