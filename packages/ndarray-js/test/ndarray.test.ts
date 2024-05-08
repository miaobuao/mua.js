import { describe, it } from 'vitest'

import { NdArray, dtype } from '..'
import { getStrides } from '../src/lib'

describe('ndarray', () => {
  it('shape', async ({ expect }) => {
    const a = new NdArray([
      [ 1, 2, 3 ],
      [ 2, 3, 4 ],
      [ 6, 6, 6 ],
    ], {
      dtype: dtype.int32,
      shape: [ 3, 3 ],
    })
    expect(a.shape).toEqual([ 3, 3 ])
  })
  it('stride', async ({ expect }) => {
    const shape1 = [ 4, 5 ]
    const strides1 = getStrides(shape1)
    expect(strides1).toEqual([ 5, 1 ])

    const shape2 = [ 4, 5, 6 ]
    const strides2 = getStrides(shape2)
    expect(strides2).toEqual([ 30, 6, 1 ])
  })

  it('get', async ({ expect }) => {
    const a = new NdArray([
      [ 1, 2, 3 ],
      [ 2, 3, 4 ],
      [ 6, 6, 6 ],
    ], { dtype: dtype.int32 })
    const b = a.slice(1)
    expect(Array.from(b.buffer)).toEqual([ 2, 3, 4 ])
    const c = new NdArray([
      [ [ 1, 2, 3 ], [ 2, 3, 4 ] ],
      [ [ 1, 2, 3 ], [ 2, 3, 4 ] ],
    ], { dtype: dtype.int32 })
    expect(c.shape).toEqual([ 2, 2, 3 ])
    const d = c.slice(0)
    expect(d.shape).toEqual([ 2, 3 ])
    const e = c.slice(0, 1)
    expect(e.shape).toEqual([ 3 ])
    const f = d.slice(0)
    expect(f.shape).toEqual([ 3 ])
  })

  it('reshape', async ({ expect }) => {
    const a = new NdArray([ [ 1, 2, 3 ], [ 2, 3, 4 ] ], { dtype: dtype.int32 })
    const b = a.reshape(-1, 2)
    expect(b.shape).toEqual([ 3, 2 ])
    const c = a.reshape(1, -1)
    expect(c.shape).toEqual([ 1, 6 ])
    const d = a.reshape(1, 2, 3)
    expect(d.toArray()).toEqual([ [ [ 1, 2, 3 ], [ 2, 3, 4 ] ] ])
  })

  it('transpose', async ({ expect }) => {
    const a = new NdArray([
      [ 1, 2, 3 ],
      [ 4, 5, 6 ],
    ], { dtype: dtype.int32 })
    expect(a.toArray()).toEqual([ [ 1, 2, 3 ], [ 4, 5, 6 ] ])
    const b = a.T
    expect(b.shape).toEqual([ 3, 2 ])
    expect(b.toArray()).toEqual([ [ 1, 4 ], [ 2, 5 ], [ 3, 6 ] ])
    const c = a.reshape(1, -1).T
    expect(c.toArray()).toEqual([ [ 1 ], [ 2 ], [ 3 ], [ 4 ], [ 5 ], [ 6 ] ])
    const d = b.reshape(-1, 1)
    expect(d.toArray()).toEqual([ [ 1 ], [ 4 ], [ 2 ], [ 5 ], [ 3 ], [ 6 ] ])
  })
})
