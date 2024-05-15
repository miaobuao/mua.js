import { normal } from '@mua/matops'
import { flatten, isInteger, range, sum } from 'lodash-es'

import * as _dtype from './dtype'
import { arrayInit, getArrayShape, getStrides } from './lib'
import { add } from './ops/add'
import { argmax } from './ops/argmax'
import { exp } from './ops/exp'
import { log } from './ops/log'
import { matmul } from './ops/matmul'
import { mul } from './ops/mul'
import { pow } from './ops/pow'
import { relu } from './ops/relu'
import { sigmod } from './ops/sigmod'
import { softmax } from './ops/softmax'
import { sub } from './ops/sub'
import { tanh } from './ops/tanh'

export { dtype } from './dtype'

type Many<T> = T | Many<T>[]

export class NdArray<Type extends _dtype.dtype = _dtype.dtype> {
  buffer: _dtype.DataTypeToArrayType[Type]
  readonly dtype: _dtype.dtype
  readonly shape: number[]
  readonly strides: number[]

  constructor(
    array: ArrayLike<Many<number>> | _dtype.DataTypeToArrayType[Type],
    {
      dtype,
      shape,
      strides,
    }: {
      dtype?: Type
      shape?: number[]
      strides?: number[]
      discontinuous?: boolean
    } = {},
  ) {
    if (Array.isArray(array)) {
      this.dtype = dtype || _dtype.dtype.float32
      this.shape = shape || getArrayShape(array)
      this.buffer = arrayInit(
        this.shape.reduce((a, b) => a * b),
        this.dtype,
      ) as any
      this.buffer.set(flatten(array))
    }
    else {
      this.dtype
        = dtype
        || _dtype.getDType(array as _dtype.DataTypeToArrayType[_dtype.dtype])
      this.shape = shape || [ array.length ]
      this.buffer = arrayInit(array.length, this.dtype) as any
      this.buffer.set(array as any)
    }
    this.strides = strides || getStrides(this.shape)
  }

  static arange(opt: {
    start?: number
    stop: number
    step?: number
    dtype?: _dtype.dtype
    shape?: number[]
  }) {
    let { start, stop, step, dtype, shape } = opt
    start ||= 0
    step ||= 1
    dtype ||= _dtype.dtype.float32
    const len = Math.floor((stop - start) / step)
    if (len <= 0)
      throw new Error(`stop=${stop} must be larger than start=${start}`)
    const buf = arrayInit(len, dtype)
    for (let i = 0; i < len; ++i) buf[i] = start + i * step
    return new NdArray(buf, {
      shape: shape || [ len ],
      dtype,
    })
  }

  static random<TDtype extends _dtype.dtype>(shape: number[], dtype?: TDtype) {
    const len = shape.reduce((a, b) => a * b)
    const buf = arrayInit(len, dtype || _dtype.dtype.float32)
    for (let i = 0; i < buf.length; ++i) buf[i] = Math.random()
    return new NdArray(buf, {
      shape,
      dtype,
    })
  }

  static normal<TDtype extends _dtype.dtype = _dtype.dtype.float32>(
    shape: number[],
    { mean, std, dtype }: { mean?: number, std?: number, dtype?: TDtype } = {},
  ) {
    const len = shape.reduce((a, b) => a * b)
    const rand = normal(len, mean || 0, std || 1)
    const buf = arrayInit(len, dtype || _dtype.dtype.float32)
    buf.set(rand)
    return new NdArray(buf, {
      shape,
      dtype,
    })
  }

  static randn(shape: number[], dtype: _dtype.dtype = _dtype.dtype.float32) {
    return NdArray.normal(shape, { dtype })
  }

  static ones<TDtype extends _dtype.dtype>(shape: number[], dtype?: TDtype) {
    dtype ||= _dtype.dtype.float32 as TDtype
    const buf = arrayInit(
      shape.reduce((a, b) => a * b),
      dtype,
    ).fill(1)
    return new NdArray(buf, { shape, dtype })
  }

  static onesLike<T extends _dtype.dtype>(t: NdArray, dtype: T) {
    return NdArray.ones(t.shape, dtype)
  }

  static zeros<TDtype extends _dtype.dtype>(shape: number[], dtype?: TDtype) {
    dtype ||= _dtype.dtype.float32 as TDtype
    const buf = arrayInit(
      shape.reduce((a, b) => a * b),
      dtype,
    ).fill(0)
    return new NdArray(buf, { shape, dtype })
  }

  static zerosLike<T extends _dtype.dtype>(t: NdArray, dtype: T) {
    return NdArray.zeros(t.shape, dtype)
  }

  map(fn: (d: number) => number): NdArray<Type> {
    return new NdArray(this.buffer.map(fn), {
      shape: this.shape,
      strides: this.strides,
      dtype: this.dtype as Type,
    })
  }

  slice(...indices: number[]): NdArray<Type> {
    const shape = this.shape.slice(indices.length)
    const length = shape.reduce((a, b) => a * b, 1)
    let stride = 0
    for (let i = 0; i < indices.length; ++i)
      stride += indices[i]! * this.strides[i]!
    return new NdArray(this.buffer.subarray(stride, stride + length) as any, {
      shape,
      dtype: this.dtype as Type,
    })
  }

  reshape(size: number[]): typeof this {
    const newAxisCount = size.filter(d => d === -1).length
    if (newAxisCount === 0) {
      if (size.reduce((a, b) => a * b, 1) !== this.buffer.length) {
        throw new Error(
          'The new shape is not compatible with the original shape.',
        )
      }
      return new NdArray(this.buffer, {
        shape: size,
        dtype: this.dtype as Type,
      }) as typeof this
    }
    else if (newAxisCount > 1) {
      throw new Error('NdArray: can only specify one unknown dimension.')
    }
    let knownSize = 1
    for (let i = 0; i < size.length; ++i) {
      const s = size[i]!
      if (s > 0) {
        knownSize *= s
      }
      else if (s < -1) {
        throw new Error(
          'NdArray: unknown dimension size must be positive or -1.',
        )
      }
    }
    const newAxis = this.buffer.length / knownSize
    if (!isInteger(newAxis)) {
      throw new Error(
        `NdArray: cannot reshape array of size ${this.buffer.length} into shape [${size}]`,
      )
    }
    const newShape = size.map(d => (d === -1 ? newAxis : d))

    return new NdArray(this.buffer, {
      shape: newShape,
      dtype: this.dtype as Type,
    }) as typeof this
  }

  toArray(): ArrayLike<Many<number>> {
    const buf = this.buffer
    const shape = this.shape
    const strides = this.strides
    function dfs(offset: number, axis: number) {
      return range(shape[axis]!).map((i) => {
        const acc = offset + i * strides[axis]!
        if (axis === shape.length - 1)
          return buf[acc]
        return dfs(acc, axis + 1)
      })
    }
    return dfs(0, 0)
  }

  permute(axis: number[]) {
    const shape = axis.map(i => this.shape[i]!)
    const strides = axis.map(i => this.strides[i]!)
    const res: _dtype.DataTypeToArrayType[Type] = arrayInit(
      this.length,
      this.dtype,
    ) as any
    const buf = this.buffer
    let cnt = 0
    function dfs(offset: number, axis: number) {
      for (let i = 0; i < shape[axis]!; ++i) {
        const acc = offset + i * strides[axis]!
        if (axis === shape.length - 1) {
          res[cnt++] = buf[acc] as any
          continue
        }
        dfs(acc, axis + 1)
      }
    }
    dfs(0, 0)
    return new NdArray(res, {
      shape,
      dtype: this.dtype as Type,
    })
  }

  flatten() {
    return this.reshape([ this.shape.reduce((a, b) => a * b) ])
  }

  set(indexes: number[], value: NdArray) {
    const offset = sum(
      this.strides
        .slice(0, indexes.length)
        .map((stride, index) => stride * indexes[index]!, 1),
    )
    this.buffer.set(value.buffer, offset)
  }

  get length() {
    return this.buffer.length
  }

  get T() {
    return this.permute(range(this.shape.length).reverse())
  }

  get value() {
    return this.buffer
  }

  async add(rhs: NdArray<_dtype.dtype> | number) {
    return add(this, rhs)
  }

  async argmax() {
    // if (this.shape.length !== 1)
    //   throw new Error('NdArray: argmax only support 1D array')
    return argmax(this)
  }

  async exp() {
    return exp(this)
  }

  async log(base = Math.E) {
    return log(this, base)
  }

  async matmul(rhs: NdArray<_dtype.dtype>) {
    return matmul(this, rhs)
  }

  async mul(rhs: NdArray<_dtype.dtype> | number) {
    return mul(this, rhs)
  }

  async pow(exponent: number) {
    return pow(this, exponent)
  }

  async relu() {
    return relu(this)
  }

  async sigmod() {
    return sigmod(this)
  }

  async softmax(dim?: number) {
    return softmax(this, dim)
  }

  async sub(rhs: NdArray<_dtype.dtype> | number) {
    return sub(this, rhs)
  }

  async sum() {
    let res = this.buffer[0]!
    for (let i = 1; i < this.buffer.length; ++i) res += this.buffer[i]!
    return res
  }

  async tanh() {
    return tanh(this)
  }
}

export type MaybePromise<T> = T | Promise<T>
