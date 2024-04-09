import type { MathCollection } from 'mathjs'

import { multiply as _mul } from 'mathjs'

export async function mulScalar(a: MathCollection, b: number) {
  return _mul(a, b)
}
