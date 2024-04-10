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
    const losses: any[] = []
    for (const e of range(10)) {
      sgd.resetGrad()
      const z = await linear.forward(x)
      const loss = await l2loss.forward(z, y)
      await loss.backward()
      losses.push({ epoch: e, loss: await loss.sum() })
      await sgd.step()
    }
    // eslint-disable-next-line no-console
    console.table(losses)
  })
})
