import { flatten, isInteger, range } from 'lodash-es'

import * as _dtype from './dtype'
import { getArrayShape, getStrides } from './lib'

export { dtype } from './dtype'

type Many<T> = T | Many<T>[]

export class NdArray<Type extends _dtype.dtype = _dtype.dtype.float32> {
  buffer: _dtype.DataTypeToArrayType[Type]
  readonly dtype: _dtype.dtype
  readonly shape: number[]
  readonly strides: number[]
  private readonly discontinuous: boolean

  constructor(
    array: ArrayLike<Many<number>> | _dtype.DataTypeToArrayType[Type],
    { dtype, shape, strides, discontinuous }: {
      dtype?: Type
      shape?: number[]
      strides?: number[]
      discontinuous?: boolean
    } = {},
  ) {
    this.discontinuous = discontinuous ?? false

    if (Array.isArray(array)) {
      this.dtype = dtype || _dtype.dtype.float32
      this.shape = shape || getArrayShape(array)
      this.buffer = arrayInit(this.shape.reduce((a, b) => a * b), this.dtype) as any
      const flattenArray = flatten(array)
      for (let i = 0; i < flattenArray.length; i++)
        this.buffer[i] = flattenArray[i]
    }
    else {
      this.buffer = array as any
      this.shape = shape || [ this.buffer.length ]
      this.dtype = dtype || _dtype.getDType(array)
    }
    this.strides = strides || getStrides(this.shape)
  }

  slice(...indices: number[]): NdArray<Type> {
    const shape = this.shape.slice(indices.length)
    const length = shape.reduce((a, b) => a * b, 1)
    let stride = 0
    for (let i = 0; i < indices.length; ++i)
      stride += indices[i]! * this.strides[i]!
    return new NdArray(
      this.buffer.subarray(stride, stride + length) as any,
      { shape, dtype: this.dtype as Type },
    )
  }

  reshape(...size: number[]): NdArray<Type> {
    const newAxisCount = size.filter(d => d === -1).length
    if (newAxisCount === 0) {
      if (size.reduce((a, b) => a * b, 1) !== this.buffer.length)
        throw new Error('The new shape is not compatible with the original shape.')
      return new NdArray(this.buffer, { shape: size, dtype: this.dtype as Type })
    }
    else if (newAxisCount > 1) {
      throw new Error('NdArray: can only specify one unknown dimension.')
    }
    let knownSize = 1
    for (let i = 0; i < size.length; ++i) {
      const s = size[i]!
      if (s > 0)
        knownSize *= s
      else if (s < -1)
        throw new Error('NdArray: unknown dimension size must be positive or -1.')
    }
    const newAxis = this.buffer.length / knownSize
    if (!isInteger(newAxis))
      throw new Error(`NdArray: cannot reshape array of size ${this.buffer.length} into shape [${size}]`)
    const newShape = size.map(d => d === -1 ? newAxis : d)
    if (this.discontinuous)
      return new NdArray(this.value, { shape: newShape, dtype: this.dtype as Type })
    return new NdArray(this.buffer, { shape: newShape, dtype: this.dtype as Type })
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

  permute(...axis: number[]) {
    const shape = axis.map(i => this.shape[i]!)
    const strides = axis.map(i => this.strides[i]!)
    return new NdArray(this.buffer, {
      shape,
      strides,
      dtype: this.dtype as Type,
      discontinuous: true,
    })
  }

  get length() {
    return this.buffer.length
  }

  get T() {
    return new NdArray(this.buffer, {
      shape: this.shape.slice().reverse(),
      strides: this.strides.slice().reverse(),
      dtype: this.dtype as Type,
      discontinuous: true,
    })
  }

  get value() {
    const res: _dtype.DataTypeToArrayType[Type] = arrayInit(this.length, this.dtype) as any
    const buf = this.buffer
    const shape = this.shape
    const strides = this.strides
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
    return (dfs(0, 0), res)
  }
}

function arrayInit(size: number, dtype: _dtype.dtype) {
  const constructor = _dtype.DataTypeToArrayConstructor[dtype]
  const sharedBuffer = new SharedArrayBuffer(size * constructor.BYTES_PER_ELEMENT)
  return new constructor(sharedBuffer)
}
