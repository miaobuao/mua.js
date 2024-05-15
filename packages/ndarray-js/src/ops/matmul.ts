import type { DataTypeToArrayType } from '../dtype'

import { matmul_f32, matmul_f64, matmul_i16, matmul_i32, matmul_i8, matmul_u16, matmul_u32, matmul_u8 } from '@mua/matops'
import { range } from 'lodash-es'

import { dtype } from '../dtype'
import { arrayInit } from '../lib'
import { NdArray } from '../ndarray'

const WASM = true

interface NdArrayMetadata {
  shape: number[]
  strides: number[]
  dtype: dtype
}

function wasm_procedure<T extends DataTypeToArrayType[dtype]>(a: T, am: NdArrayMetadata, b: T, bm: NdArrayMetadata): T {
  if (am.dtype !== bm.dtype)
    throw new Error('data type mismatch')

  switch (am.dtype) {
    case dtype.float32: return matmul_f32(a as any, { shape: am.shape, strides: am.strides }, b as any, { shape: bm.shape, strides: bm.strides }) as any
    case dtype.float64: return matmul_f64(a as any, { shape: am.shape, strides: am.strides }, b as any, { shape: bm.shape, strides: bm.strides }) as any
    case dtype.int16: return matmul_i16(a as any, { shape: am.shape, strides: am.strides }, b as any, { shape: bm.shape, strides: bm.strides }) as any
    case dtype.int32: return matmul_i32(a as any, { shape: am.shape, strides: am.strides }, b as any, { shape: bm.shape, strides: bm.strides }) as any
    // case dtype.int64: return matmul_i64(a as any, { shape: am.shape, strides: am.strides }, b as any, { shape: bm.shape, strides: bm.strides }) as any
    case dtype.int8: return matmul_i8(a as any, { shape: am.shape, strides: am.strides }, b as any, { shape: bm.shape, strides: bm.strides }) as any
    case dtype.uint16: return matmul_u16(a as any, { shape: am.shape, strides: am.strides }, b as any, { shape: bm.shape, strides: bm.strides }) as any
    case dtype.uint32: return matmul_u32(a as any, { shape: am.shape, strides: am.strides }, b as any, { shape: bm.shape, strides: bm.strides }) as any
    // case dtype.uint64: return matmul_u64(a as any, { shape: am.shape, strides: am.strides }, b as any, { shape: bm.shape, strides: bm.strides }) as any
    case dtype.uint8: return matmul_u8(a as any, { shape: am.shape, strides: am.strides }, b as any, { shape: bm.shape, strides: bm.strides }) as any
    default: throw new Error(`Unsupported dtype: ${am.dtype}`)
  }
}

export async function matmul<T extends dtype = any>(a: NdArray<T>, b: NdArray<T>): Promise<NdArray<T>> {
  if (a.shape.length !== 2 && b.shape.length !== 2)
    throw new Error('matmul only support 2D matrix')

  const shape = [ a.shape[0]!, b.shape[1]! ]
  if (WASM) {
    const res = wasm_procedure(a.value, { shape: a.shape, strides: a.strides, dtype: a.dtype }, b.value, { shape: b.shape, strides: b.strides, dtype: b.dtype })
    return new NdArray(res, { shape })
  }
  const res = arrayInit(a.shape[0]! * b.shape[1]!, a.dtype)
  const bufferA = a.value
  const bufferB = b.value
  let cnt = 0
  for (let i = 0; i < shape[0]!; ++i) {
    for (let j = 0; j < shape[1]!; ++j) {
      const sum = range(a.shape[1]!).map(k => (bufferA[i * a.shape[1]! + k] as any) * (bufferB[k * b.shape[1]! + j] as any)).reduce((a, b) => a + b, 0)
      res[cnt++] = sum
    }
  }
  return new NdArray(res as any, { shape })
}
