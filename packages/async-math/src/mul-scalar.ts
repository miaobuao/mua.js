import type { NdArrayNumberCell } from './ndarray'
import type { MaybePromise } from '@mua/common'

import { multiply as _mul } from 'mathjs'

import { NdArray } from './ndarray'

export async function mulScalar(
  a: MaybePromise<NdArray<number> | NdArrayNumberCell>,
  b: MaybePromise<number>,
) {
  const [ x, y ] = await Promise.all([ a, b ])
  return new NdArray(
    _mul(NdArray.toValue(x) as number[], y) as number[],
  )
}
