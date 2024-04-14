import { reshape } from 'mathjs'

import { NdArray } from './ndarray'

export function Reshape(size: number[]) {
  return <T extends Array<any> | NdArray>(x: T) => x instanceof NdArray ? x.reshape(size) : reshape(x, size) as T
}
