import { assert, describe, it } from 'vitest'

import { getConvSize, getPoolSize } from '../src'

describe('common/conv', () => {
  it.concurrent('conv size', async () => {
    assert.deepEqual(getConvSize([ 10, 10 ], [ 3, 3 ]), [ 8, 8 ])
    assert.deepEqual(getConvSize([ 10, 10 ], [ 3, 3 ], 2), [ 4, 4 ])
    assert.deepEqual(getConvSize([ 10, 10 ], [ 3, 3 ], 2, 1), [ 5, 5 ])
  })

  it.concurrent('pool size', async () => {
    assert.deepEqual(getPoolSize([ 9, 9 ], [ 3, 3 ]), [ 3, 3 ])
    assert.deepEqual(getPoolSize([ 10, 10 ], [ 3, 3 ], 2), [ 4, 4 ])
    assert.deepEqual(getPoolSize([ 10, 10 ], [ 3, 3 ], 2, 1), [ 5, 5 ])
  })
})
