import { ones } from '@mua/tensor'
import { describe, it } from 'vitest'

import { Conv1d, Conv2d, Linear } from '../src'

describe('module', () => {
  it.concurrent('linear', async ({ expect }) => {
    const linear = new Linear(10, 30)
    const x = ones(20, 10)
    const res = await linear.forward(x)
    expect(await linear.weight.shape).toEqual([ 10, 30 ])
    expect(await res.shape).toEqual([ 20, 30 ])
  })

  it.concurrent('conv1d', async ({ expect }) => {
    const conv1d = new Conv1d(3, 8, 3)
    const x = ones(20, 3)
    const res = await conv1d.forward(x)
    expect(await res.shape).toEqual([ 18, 8 ])
  })

  it.concurrent('conv2d', async ({ expect }) => {
    const conv2d = new Conv2d(3, 8, 3)
    const x = ones(32, 32, 3)
    const res = await conv2d.forward(x)
    expect(await res.shape).toEqual([ 30, 30, 8 ])
  })
})
