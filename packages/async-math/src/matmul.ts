import type { NdArrayNumberCell } from './ndarray'
import type { MaybePromise } from '@mua/common'

import { multiply as _mul } from 'mathjs'

import { NdArray } from './ndarray'

export async function matmul(
  a: MaybePromise<NdArray<number> | NdArrayNumberCell>,
  b: MaybePromise<NdArray<number> | NdArrayNumberCell>,
) {
  const [ _a, _b ] = await Promise.all([ a, b ])
  return new NdArray(
    _mul(NdArray.toValue(_a) as number[], NdArray.toValue(_b) as number[]),
  )
}
