import { ones } from '@mua/tensor'
import { describe, it } from 'vitest'

import { Linear } from '../src'

describe('module', () => {
  it.concurrent('linear', async ({ expect }) => {
    const linear = new Linear(10, 30)
    const x = ones(20, 10)
    const res = await linear.forward(x)
    expect(await linear.weight.shape).toEqual([ 10, 30 ])
    expect(await res.shape).toEqual([ 20, 30 ])
  })
})
