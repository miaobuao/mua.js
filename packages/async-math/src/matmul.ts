import type { NdArrayNumberCell } from './ndarray'

import { multiply as _mul } from 'mathjs'

import { NdArray } from './ndarray'

export async function matmul(
  a: NdArray<number> | NdArrayNumberCell,
  b: NdArray<number> | NdArrayNumberCell,
) {
  return new NdArray(
    _mul(NdArray.toValue(a) as number[], NdArray.toValue(b) as number[]),
  )
}
