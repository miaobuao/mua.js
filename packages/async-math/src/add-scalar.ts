import { type MathCollection, add as _add } from 'mathjs'

export async function addScalar(a: MathCollection, b: number) {
  return _add(a, b)
}
