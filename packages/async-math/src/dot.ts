import type { NdArrayNumberCell } from './ndarray'
import type { MaybePromise } from '@mua/common'
import type { MathNumericType } from 'mathjs'

import { dotMultiply as _dot } from 'mathjs'

import { NdArray } from './ndarray'

export async function dot(
  a: MaybePromise<NdArray<number> | NdArrayNumberCell>,
  b: MaybePromise< NdArray<number> | NdArrayNumberCell | MathNumericType>,
) {
  const [ x, y ] = await Promise.all([ a, b ])
  return new NdArray(
    _dot(NdArray.toValue(x) as number[], NdArray.toValue(y) as number[]),
  )
}
