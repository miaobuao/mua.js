import { type MathType, add as _add } from 'mathjs'

export async function add(a: MathType, b: MathType) {
  return _add(a, b)
}
