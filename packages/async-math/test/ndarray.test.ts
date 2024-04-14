import { random } from 'mathjs'
import { describe, it } from 'vitest'

import { NdArray } from '../src'

describe('ndarray', () => {
  it('ndarray', async ({ expect }) => {
    const a = new NdArray([ [ 1, 2, 3 ], [ 2, 3, 4 ], [ 6, 6, 6 ] ])
    expect(a.shape).toEqual([ 3, 3 ])

    const b = await a.permute([ 1, 0 ])
    expect(b.value).toEqual([
      [ 1, 2, 6 ],
      [ 2, 3, 6 ],
      [ 3, 4, 6 ],
    ])

    const c = new NdArray(random([ 12, 23, 32, 33 ]))
    const d = await c.permute([ 2, 0, 3, 1 ])
    expect(d.shape).toEqual([ 32, 12, 33, 23 ])
    expect((await c.T).shape).toEqual([ 33, 32, 23, 12 ])
  })
})
