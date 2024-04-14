import { describe, it } from 'vitest'

import { conv1d, conv2d, random } from '../src'

describe('conv', () => {
  it('conv1d', async ({ expect }) => {
    const a = random([ 10, 16 ])
    const k = random([ 3, 16, 32 ])

    const res1 = await conv1d(a, k)
    expect(await res1.shape).toEqual([ 8, 32 ])

    const res2 = await conv1d(a, k, { padding: 1 })
    expect(await res2.shape).toEqual([ 10, 32 ])

    const res3 = await conv1d(a, k, { stride: 2 })
    expect(await res3.shape).toEqual([ 4, 32 ])

    const res4 = await conv1d(a, k, { padding: 1, stride: 2 })
    expect(await res4.shape).toEqual([ 5, 32 ])
  })

  it('conv2d', async ({ expect }) => {
    const a = random([ 10, 5, 3 ])
    const k = random([ 3, 3, 3, 16 ])

    const res1 = await conv2d(a, k)
    expect(await res1.shape).toEqual([ 8, 3, 16 ])
  })
})
