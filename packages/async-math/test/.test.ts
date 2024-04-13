import { random } from 'mathjs'
import { describe, it } from 'vitest'

import { conv1d, matrix } from '../src'

describe('async math', () => {
  it.concurrent('test conv1d', async ({ expect }) => {
    const x = matrix(random([ 10, 32 ]))
    const kernel = matrix(random([ 3, 32, 16 ]))

    const res1 = await conv1d(x, kernel, { padding: 1 })
    expect(res1.size()).toEqual([ 10, 16 ])

    const res2 = await conv1d(x, kernel, { stride: 2 })
    expect(res2.size()).toEqual([ 4, 16 ])

    const res3 = await conv1d(x, kernel, { padding: 1, stride: 2 })
    expect(res3.size()).toEqual([ 5, 16 ])
  })
})
