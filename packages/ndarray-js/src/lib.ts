import { DataTypeToArrayConstructor, type DataTypeToArrayType, type dtype, getDType } from './dtype'

export function getArrayShape(array: ArrayLike<any>) {
  const shape: number[] = []
  while (Array.isArray(array)) {
    shape.push(array.length)
    array = array[0]
  }
  return shape
}

export function getStrides(shape: number[]) {
  return shape.map((_, i) => shape.slice(i + 1, shape.length).reduce((a, b) => a * b, 1))
}

export function ndIdxToOffset(index: number[], stride: number[]) {
  return index.map((d, axis) => d * stride[axis]!).reduce((a, b) => a + b, 0)
}

export function concatTypedArray<T extends DataTypeToArrayType[dtype]>(args: T[]): T {
  let len = 0
  for (const d of args)
    len += d.length
  const type = getDType(args[0]!)
  const res = arrayInit(len, type)
  let offset = 0
  for (const d of args) {
    res.set(d, offset)
    offset += d.length
  }
  return res as T
}

export function arrayInit(size: number, dtype: dtype) {
  const constructor = DataTypeToArrayConstructor[dtype]
  const sharedBuffer = new SharedArrayBuffer(
    size * constructor.BYTES_PER_ELEMENT,
  )
  return new constructor(sharedBuffer)
}
