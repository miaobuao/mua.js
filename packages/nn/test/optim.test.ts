/* eslint-disable no-console */
import { ones, zeros } from '@mua/tensor'
import { range } from 'lodash-es'
import { describe, it } from 'vitest'

import { L2Loss, Linear, SGD } from '../src'

describe('optim', () => {
  const l2loss = new L2Loss()
  it.concurrent('optim linear', async () => {
    const linear = new Linear(10, 30)
    const sgd = new SGD(linear.parameters(), 1e-3)
    const x = ones(20, 10)
    const y = zeros(20, 30)
    for (const e of range(20)) {
      sgd.resetGrad()
      const z = await linear.forward(x)
      const loss = await l2loss.forward(z, y)
      await loss.backward()
      console.log(e, await loss.sum())
      sgd.step()
    }
  })
})
