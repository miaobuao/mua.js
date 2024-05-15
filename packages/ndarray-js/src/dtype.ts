enum DType {
  float32 = 'float32',
  float64 = 'float64',
  int32 = 'int32',
  // int64 = 'int64',
  uint8 = 'uint8',
  uint16 = 'uint16',
  uint32 = 'uint32',
  // uint64 = 'uint64',
  int8 = 'int8',
  int16 = 'int16',
  bool = 'bool',
}

export { DType as dtype }

export const DataTypeToArrayConstructor = {
  [DType.float32]: Float32Array,
  [DType.float64]: Float64Array,
  [DType.int32]: Int32Array,
  // [DType.int64]: BigInt64Array,
  [DType.uint8]: Uint8Array,
  [DType.uint16]: Uint16Array,
  [DType.uint32]: Uint32Array,
  // [DType.uint64]: BigUint64Array,
  [DType.int8]: Int8Array,
  [DType.int16]: Int16Array,
  [DType.bool]: Uint8Array,
} as const

export interface DataTypeToArrayType {
  [DType.float32]: Float32Array
  [DType.float64]: Float64Array
  [DType.int32]: Int32Array
  // [DType.int64]: BigInt64Array
  [DType.uint8]: Uint8Array
  [DType.uint16]: Uint16Array
  [DType.uint32]: Uint32Array
  // [DType.uint64]: BigUint64Array
  [DType.int8]: Int8Array
  [DType.int16]: Int16Array
  [DType.bool]: Uint8Array
}

export function getDType(arr: DataTypeToArrayType[DType]): DType {
  if (arr instanceof Float32Array)
    return DType.float32
  if (arr instanceof Float64Array)
    return DType.float64
  if (arr instanceof Int32Array)
    return DType.int32
  // if (arr instanceof BigInt64Array)
  //   return DType.int64
  if (arr instanceof Uint8Array)
    return DType.uint8
  if (arr instanceof Uint16Array)
    return DType.uint16
  if (arr instanceof Uint32Array)
    return DType.uint32
  // if (arr instanceof BigUint64Array)
  //   return DType.uint64
  if (arr instanceof Int8Array)
    return DType.int8
  if (arr instanceof Int16Array)
    return DType.int16
  throw new Error('Unknown array type')
}
