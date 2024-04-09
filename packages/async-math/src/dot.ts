import { type MathCollection, dotMultiply as _dot} from 'mathjs'

export async function dot(a: MathCollection, b: MathCollection) {
  return _dot(a, b)
}
