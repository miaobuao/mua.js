import type { MaybePromise, NdArray } from '../ndarray'

export async function toArray(value: MaybePromise<NdArray>) {
  return (await value).toArray()
}
