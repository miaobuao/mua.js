import type { NdArrayNumberCell } from './ndarray'

import { add as _add } from 'mathjs'

import { NdArray } from './ndarray'

export async function add(
  a: NdArray<number> | NdArrayNumberCell,
  b: NdArray<number> | NdArrayNumberCell | number,
) {
  return new NdArray(
    _add(NdArray.toValue(a) as number[], NdArray.toValue(b) as number[]),
  )
}
