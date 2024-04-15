import type { NdArrayNumberCell } from './ndarray'
import type { MaybePromise } from '@mua/common'

import { NdArray, mapElementForArray } from './ndarray'

export async function negative(value: MaybePromise<NdArray | NdArrayNumberCell[]>) {
  const v = await value
  if (v instanceof NdArray)
    return v.mapElement(d => -d)
  return new NdArray(mapElementForArray(v, d => -d))
}
