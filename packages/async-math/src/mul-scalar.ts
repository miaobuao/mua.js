import type { NdArrayNumberCell } from './ndarray'

import { multiply as _mul } from 'mathjs'

import { NdArray } from './ndarray'

export async function mulScalar(
  a: NdArray<number> | NdArrayNumberCell,
  b: number,
) {
  return new NdArray(
    _mul(NdArray.toValue(a) as number[], b) as number[],
  )
}
