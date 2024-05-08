import { ones } from '@mua/tensor'
import { describe, it } from 'vitest'

import { Conv1d, Conv2d, Linear } from '../src'

describe('module', () => {
  it.concurrent('linear', async ({ expect }) => {
    const linear = new Linear(10, 30)
    const x = ones(new Uint32Array([ 20, 10 ]))
    const res = await linear.forward(x)
    expect(await linear.weight.shape).toEqual(new Uint32Array([ 10, 30 ]))
    expect(await res.shape).toEqual(new Uint32Array([ 20, 30 ]))
  })

  it.concurrent('conv1d', async ({ expect }) => {
    const conv1d = new Conv1d(3, 8, 3)
    const x = ones(new Uint32Array([ 20, 3 ]))
    const res = await conv1d.forward(x)
    expect(await res.shape).toEqual(new Uint32Array([ 18, 8 ]))
  })

  it.concurrent('conv2d', async ({ expect }) => {
    const conv2d = new Conv2d(1, 8, 3)
    const x = ones(new Uint32Array([ 28, 28, 1 ]))
    const res = await conv2d.forward(x)
    expect(await res.shape).toEqual(new Uint32Array([ 26, 26, 8 ]))
  })
})
