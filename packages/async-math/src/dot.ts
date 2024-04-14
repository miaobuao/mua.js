import type { NdArrayNumberCell } from './ndarray'
import type { MathNumericType } from 'mathjs'

import { dotMultiply as _dot } from 'mathjs'

import { NdArray } from './ndarray'

export async function dot(
  a: NdArray<number> | NdArrayNumberCell,
  b: NdArray<number> | NdArrayNumberCell | MathNumericType,
) {
  return new NdArray(
    _dot(NdArray.toValue(a) as number[], NdArray.toValue(b) as number[]),
  )
}
