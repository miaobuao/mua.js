import { describe, expect, it } from 'vitest'

import { NdArray, dtype } from '../src/ndarray'
import { im2col } from '../src/ops'

describe('ndarray:ops', () => {
  it.concurrent('matmul', async ({ expect }) => {
    const a = new NdArray([ [ 1, 2, 3 ], [ 2, 3, 4 ] ], { dtype: dtype.float32 })
    const b = a.T
    const c = await a.matmul(b)
    expect(c.toArray()).toEqual([ [ 14, 20 ], [ 20, 29 ] ])
  })

  it.concurrent('argmax', async () => {
    const a = new NdArray([ 1, 2, 3, 4, 66, 2, 3 ])
    const b = await a.argmax()
    expect(b).toEqual(4)
  })

  it.concurrent('softmax', async ({ expect }) => {
    const a = new NdArray([ 1, 2, 3, 4 ])
    expect((await a.softmax()).toArray()).toEqual([ 0.032058604061603546, 0.08714432269334793, 0.23688282072544098, 0.6439142227172852 ])
  })

  it.concurrent('im2col', () => {
    // for 2D
    const im0 = NdArray.arange({
      stop: 10,
      shape: [ 5, 2 ],
    })
    const res0 = im2col(im0, [ 3 ], 1)
    expect(res0.toArray()).toEqual([ [ 0, 1, 2, 3, 4, 5 ], [ 2, 3, 4, 5, 6, 7 ], [ 4, 5, 6, 7, 8, 9 ] ])
    expect(res0.shape).toEqual([ 3, 6 ])

    // for 3D
    const im1 = NdArray.arange({
      stop: 5 * 5 * 3,
      shape: [ 5, 5, 3 ],
    })
    const res1 = im2col(im1, [ 3, 3 ], 2)
    expect(res1.shape).toEqual([ 4, 27 ])
  })
})
