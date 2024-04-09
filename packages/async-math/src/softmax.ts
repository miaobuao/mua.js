import type { MathCollection } from 'mathjs'

import { divide, exp, map, max, subtract, sum } from 'mathjs'

/**
 *
 * @param a vector like [0, 1, 2, ...]
 * @returns vector like [0, 1, 2, ...]
 */
export async function softmax(a: MathCollection) {
  const c = max(a)
  a = map(a, d => exp(subtract(d, c)))
  const sumed = sum(a)
  return a.map(d => divide(d, sumed))
}
