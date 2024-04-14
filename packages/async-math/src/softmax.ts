import type { NdArrayNumberCell } from './ndarray'
import type { Matrix } from 'mathjs'

import { divide, exp, map, matrix, max, subtract, sum } from 'mathjs'

import { NdArray } from './ndarray'

export async function softmax(array: NdArray<number> | NdArrayNumberCell) {
  const value = NdArray.toValue(array)
  let a = matrix(value as number[])
  if (a.size().length > 1) {
    const c = max(a, 1) as unknown as Matrix
    a = a.map((d, [ idx ]) => exp(subtract(d, c.get([ idx! ]))))
    const sumed = sum(a, 1) as unknown as Matrix
    return a.map((d, [ idx ]) => divide(d, sumed.get([ idx! ])))
  }
  else {
    const c = max(a)
    a = map(a, d => exp(subtract(d, c)))
    const sumed = sum(a)
    return a.map(d => divide(d, sumed))
  }
}
