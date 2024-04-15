import type { NdArrayNumberCell } from './ndarray'
import type { MaybePromise } from '@mua/common'
import type { BigNumber, Complex } from 'mathjs'

import { isNil } from 'lodash-es'
import { log as _log, log2 as _log2 } from 'mathjs'

import { NdArray, mapElementForArray } from './ndarray'

export async function log2(input: MaybePromise<NdArray | NdArrayNumberCell>) {
  const x = await input
  return new NdArray(_log2(NdArray.toValue(x) as number[]))
}

export async function log(input: MaybePromise<NdArray | NdArrayNumberCell>, base?: number | BigNumber | Complex) {
  const x = await input
  return new NdArray(
    isNil(base)
      ? mapElementForArray(NdArray.toValue(x) as number[], v => _log(v))
      : mapElementForArray(NdArray.toValue(x) as number[], v => _log(v, base)),
  )
}
