/* eslint-disable no-console */

import type { Tensor } from '@mua/tensor'

import { flatten, random } from '@mua/tensor'
import { range } from 'lodash-es'
import { describe, it } from 'vitest'

import { Conv1d, Conv2d, L2Loss, Linear, Module, SGD } from '../src'

describe('optim', () => {
  const l2loss = new L2Loss()
  it.concurrent('optim linear', async () => {
    class MyModel extends Module {
      readonly linear1 = new Linear(20, 10)
      readonly linear2 = new Linear(10, 4)
      forward(input: Tensor) {
        const x1 = this.linear1.forward(input)
        return this.linear2.forward(x1)
      }
    }
    const model = new MyModel()
    const sgd = new SGD(model.parameters(), 1e-3)
    const x = random([ 10, 20 ], 0, 1)
    const y = random([ 10, 4 ], 0, 1)
    const losses: any[] = []
    for (const e of range(5)) {
      sgd.resetGrad()
      const z = await model.forward(x)
      const loss = await l2loss.forward(z, y)
      await loss.backward()
      losses.push({ epoch: e, loss: await loss.sum(), from: 'linear' })
      await sgd.step()
    }
    console.table(losses)
  })

  it.concurrent('optim conv1d', async () => {
    class MyModel extends Module {
      readonly conv1 = new Conv1d(3, 16, 3, { padding: 1 })
      readonly conv2 = new Conv1d(16, 8, 3, { padding: 1 })
      forward(input: Tensor) {
        const x1 = this.conv1.forward(input)
        return this.conv2.forward(x1)
      }
    }
    const model = new MyModel()
    const sgd = new SGD(model.parameters(), 1e-3)
    const x = random([ 5, 3 ])
    const y = random([ 5, 8 ])
    const losses: any[] = []
    for (const e of range(5)) {
      sgd.resetGrad()
      const z = await model.forward(x)
      const loss = await l2loss.forward(z, y)
      await loss.backward()
      losses.push({ epoch: e, loss: await loss.sum(), from: 'conv1d' })
      await sgd.step()
    }
    console.table(losses)
  })

  it.concurrent('optim conv2d', async () => {
    class MyModel extends Module {
      readonly conv1 = new Conv2d(4, 3, 3, { padding: 1 })
      readonly conv2 = new Conv2d(3, 1, 3, { padding: 1 })
      forward(input: Tensor) {
        const x1 = this.conv1.forward(input)
        return this.conv2.forward(x1)
      }
    }
    const model = new MyModel()
    const sgd = new SGD(model.parameters(), 1e-3)
    const x = random([ 2, 2, 4 ])
    const y = random([ 2, 2, 1 ])
    const losses: any[] = []
    for (const e of range(5)) {
      sgd.resetGrad()
      const z = await model.forward(x)
      const loss = await l2loss.forward(z, y)
      await loss.backward()
      losses.push({ epoch: e, loss: await loss.sum(), from: 'conv2d' })
      await sgd.step()
    }
    console.table(losses)
  })

  it.concurrent('optim mix nn', async () => {
    class MyModel extends Module {
      readonly conv = new Conv2d(3, 2, 3, { padding: 1 })
      readonly clf = new Linear(50, 2)
      forward(input: Tensor) {
        const x1 = this.conv.forward(input)
        const x2 = flatten(x1)
        return this.clf.forward(x2)
      }
    }
    const model = new MyModel()
    const sgd = new SGD(model.parameters(), 1e-3)
    const x = random([ 5, 5, 3 ])
    const y = random([ 1, 2 ])
    const losses: any[] = []
    for (const e of range(5)) {
      sgd.resetGrad()
      const z = await model.forward(x)
      const loss = await l2loss.forward(z, y)
      await loss.backward()
      losses.push({ epoch: e, loss: await loss.sum(), from: 'mix' })
      await sgd.step()
    }
    console.table(losses)
  })
})
