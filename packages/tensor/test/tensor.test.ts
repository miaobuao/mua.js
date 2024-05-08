import { assert, describe, it } from 'vitest'

import { isLazyMode, setLazyMode, toNdArray } from '../src'
import { add, addScalar, relu } from '../src/ops'
import { Tensor } from '../src/tensor'

describe('tensor', () => {
  const t1 = new Tensor(toNdArray([
    [ 1, 2, 3 ],
    [ 1, 2, 3 ],
  ]))
  const t2 = new Tensor(toNdArray([
    [ 1, 2, 3, 4 ],
    [ 1, 2, 3, 4 ],
    [ 1, 2, 3, 4 ],
  ]))
  it.concurrent('size', async ({ expect }) => {
    expect((await t1.shape)).toEqual(new Uint32Array([ 2, 3 ]))
    expect((await t2.shape)).toEqual(new Uint32Array([ 3, 4 ]))
  })

  it.concurrent('add', async ({ expect }) => {
    const t3 = await add(t1, t1)
    expect(t3.inputs).toEqual([ t1, t1 ])
    assert.deepEqual(await t3.toArray(), [ [ 2, 4, 6 ], [ 2, 4, 6 ] ])
  })

  it.concurrent('add scalar', async () => {
    const t3 = await addScalar(t1, 1)
    assert.deepEqual(await t3.toArray(), [ [ 2, 3, 4 ], [ 2, 3, 4 ] ])
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

  it.concurrent('relu', async () => {
    const a = new Tensor([ -1, 1, -1 ])
    const t3 = await relu(a)
    assert.deepEqual((await t3.toArray()), [ 0, 1, 0 ])
  })

  it.concurrent('argmax', async () => {
    const a = new Tensor([ 1, 4, 2, 3 ])
    assert.equal(await a.argmax(), 1)
  })
})
