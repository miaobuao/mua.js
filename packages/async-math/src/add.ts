import type { NdArrayNumberCell } from './ndarray'
import type { MaybePromise } from '@mua/common'

import { add as _add } from 'mathjs'

import { NdArray } from './ndarray'

export async function add(
  a: MaybePromise<NdArray<number> | NdArrayNumberCell>,
  b: MaybePromise<NdArray<number> | NdArrayNumberCell | number>,
) {
  const [ x, y ] = await Promise.all([ a, b ])
  return new NdArray(
    _add(NdArray.toValue(x) as number[], NdArray.toValue(y) as number[]),
  )
}
