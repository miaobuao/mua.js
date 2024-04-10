import { describe, it } from 'vitest'

import { Linear } from '../src'

describe('module', () => {
  it.concurrent('auto grad', async ({ expect }) => {
    const linear = new Linear(10, 30)
    expect(await linear.weight.shape).toEqual([ 10, 30 ])
  })
})
