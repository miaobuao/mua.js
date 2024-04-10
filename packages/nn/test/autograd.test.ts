import { Tensor, add } from '@mua/tensor'
import { assert, describe, it } from 'vitest'

import { L2Loss, Module } from '../src'

class ScaleAdd extends Module {
  readonly s: Tensor
  readonly b: Tensor
  constructor(s: number = 1, b: number = 0) {
    super()
    this.s = new Tensor([ s ])
    this.b = new Tensor([ b ])
  }

  forward(x: Tensor) {
    return add(x.mul(this.s), this.b)
  }
}

class MultiPathScaleAdd extends Module {
  constructor(
    readonly path0 = new ScaleAdd(),
    readonly path1 = new ScaleAdd(),
  ) {
    super()
  }

  forward(x: Tensor) {
    return add(
      this.path0.forward(x),
      this.path1.forward(x),
    )
  }
}

describe('auto grad', () => {
  const l2loss = new L2Loss()

  it.concurrent('test scale add', async () => {
    const sadd = new ScaleAdd(2, 2)
    const x = new Tensor([ 2 ])
    const y = await sadd.forward(x)
    assert.deepEqual(await y.toArray(), [ 6 ])
  })

  it.concurrent('test multi-path scale add', async ({ expect }) => {
    const mpath = new MultiPathScaleAdd()
    const x = new Tensor([ 2 ])
    const res = await mpath.forward(x)
    expect(mpath.parameters().length).eq(4)
    assert.deepEqual(await res.toArray(), [ 4 ])
  })

  it.concurrent('test backward', async () => {
    const mpath = new MultiPathScaleAdd()
    const x = new Tensor([ 2 ])
    const y = new Tensor([ 2 ])
    const z = await mpath.forward(x)
    const loss = await l2loss.forward(z, y)
    await loss.backward()

    assert.deepEqual(await z.toArray(), [ 4 ])
    assert.deepEqual(await loss.toArray(), [ 4 ])
    assert.deepEqual(await mpath.path0.s.gradient?.toArray(), [ 8 ])
  })
})
