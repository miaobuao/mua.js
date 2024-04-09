import type { MathCollection } from 'mathjs'

import { multiply as _mul } from 'mathjs'

export async function matmul(a: MathCollection, b: MathCollection) {
  return _mul(a, b)
}
