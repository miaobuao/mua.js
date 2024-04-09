import { assert, describe, it } from 'vitest'

import { isLazyMode, setLazyMode } from '../src'
import { add, addScalar } from '../src/ops'
import { Tensor } from '../src/tensor'

describe('tensor', () => {
  const t1 = new Tensor([
    [ 1, 2, 3 ],
    [ 1, 2, 3 ],
  ])
  const t2 = new Tensor([
    [ 1, 2, 3, 4 ],
    [ 1, 2, 3, 4 ],
    [ 1, 2, 3, 4 ],
  ])
  it.concurrent('size', async ({ expect }) => {
    expect((await t1.shape)).toEqual([ 2, 3 ])
    expect((await t2.shape)).toEqual([ 3, 4 ])
  })

  it.concurrent('add', async ({ expect }) => {
    const t3 = await add(t1, t1)
    expect(t3.inputs).toEqual([ t1, t1 ])
    assert.deepEqual((await t3.toArray()), [ [ 2, 4, 6 ], [ 2, 4, 6 ] ])
  })

  it.concurrent('add scalar', async () => {
    const t3 = await addScalar(t1, 1)
    assert.deepEqual((await t3.toArray()), [ [ 2, 3, 4 ], [ 2, 3, 4 ] ])
  })

  it.concurrent('dot', async () => {
    const t3 = await t1.dot(t1)
    assert.deepEqual((await t3.toArray()), [ [ 1, 4, 9 ], [ 1, 4, 9 ] ])
  })

  it.concurrent('mul', async () => {
    const t3 = await t1.mul(2)
    assert.deepEqual((await t3.toArray()), [ [ 2, 4, 6 ], [ 2, 4, 6 ] ])
    const t4 = await t1.mul(t3)
    assert.deepEqual((await t4.toArray()), [ [ 2, 8, 18 ], [ 2, 8, 18 ] ])
  })

  it.concurrent('lazy mode', async ({ expect }) => {
    setLazyMode(true)
    expect(isLazyMode()).toEqual(true)
    setLazyMode(false)
    expect(isLazyMode()).toEqual(false)
  })
})
