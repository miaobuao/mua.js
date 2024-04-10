import type { MathCollection, Matrix } from 'mathjs'

import { divide, exp, isArray, map, matrix, max, subtract, sum } from 'mathjs'

export async function softmax(a: MathCollection) {
  if (isArray(a))
    a = matrix(a)
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
