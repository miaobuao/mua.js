/* eslint-disable no-console */

import type { Tensor } from '@mua/tensor'

import { random, relu } from '@mua/tensor'
import { range } from 'lodash-es'
import { describe, it } from 'vitest'

import { Conv1d, Conv2d, L2Loss, Linear, Module, SGD } from '../src'

describe('optim', () => {
  const l2loss = new L2Loss()
  const epoch = 5
  it.concurrent('optim linear', async () => {
    class MyModel extends Module {
      readonly linear1 = new Linear(100, 10)
      readonly linear2 = new Linear(10, 4)
      forward(input: Tensor) {
        let x = this.linear1.forward(input)
        x = relu(x)
        x = this.linear2.forward(x)
        return x
      }
    }
    const model = new MyModel()
    const sgd = new SGD(model.parameters(), 1e-3)
    const x = random([ 20, 100 ])
    const y = random([ 20, 4 ])
    const losses: any[] = []
    for (const e of range(epoch)) {
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
      readonly conv1 = new Conv1d(10, 16, 3) // out: [18, 16]
      readonly conv2 = new Conv1d(16, 4, 3) // out: [16, 4]
      forward(input: Tensor) {
        let x = this.conv1.forward(input)
        x = this.conv2.forward(x)
        x = relu(x)
        return x
      }
    }
    const model = new MyModel()
    const sgd = new SGD(model.parameters(), 1e-3)
    const x = random([ 20, 10 ])
    const y = random([ 16, 4 ])
    const losses: any[] = []
    for (const e of range(epoch)) {
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
      readonly conv1 = new Conv2d(4, 3, 3) // out: [26, 26, 3]
      readonly conv2 = new Conv2d(3, 1, 3) // out: [24, 24, 1]
      forward(input: Tensor) {
        let x = this.conv1.forward(input)
        x = this.conv2.forward(x)
        x = relu(x)
        return x
      }
    }
    const model = new MyModel()
    const sgd = new SGD(model.parameters(), 1e-3)
    const x = random([ 28, 28, 4 ])
    const y = random([ 24, 24, 1 ])
    const losses: any[] = []
    for (const e of range(epoch)) {
      sgd.resetGrad()
      const z = await model.forward(x)
      const loss = await l2loss.forward(z, y)
      await loss.backward()
      losses.push({ epoch: e, loss: await loss.sum(), from: 'conv2d' })
      await sgd.step()
    }
    console.table(losses)
  })

  // it.concurrent('optim mix nn', async () => {
  //   class MyModel extends Module {
  //     readonly conv = new Conv2d(3, 8, 3, { stride: 2 })
  //     readonly clf = new Linear(13 * 13 * 8, 10)
  //     forward(input: Tensor) {
  //       const x1 = this.conv.forward(input)
  //       const x2 = flatten(x1)
  //       return this.clf.forward(x2)
  //     }
  //   }
  //   const model = new MyModel()
  //   const sgd = new SGD(model.parameters(), 1e-3)
  //   const x = random([ 28, 28, 3 ])
  //   const y = random([ 1, 10 ])
  //   const losses: any[] = []
  //   for (const e of range(epoch)) {
  //     sgd.resetGrad()
  //     const z = await model.forward(x)
  //     const loss = await l2loss.forward(z, y)
  //     await loss.backward()
  //     losses.push({ epoch: e, loss: await loss.sum(), from: 'mix' })
  //     await sgd.step()
  //   }
  //   console.table(losses)
  // })

  // it.concurrent('optim cross-entropy', async () => {
  //   class MyModel extends Module {
  //     readonly conv1 = new Conv2d(3, 8, 3, { stride: 3 })
  //     readonly conv2 = new Conv2d(8, 1, 3, { })
  //     readonly clf = new Linear(196, 10)
  //     forward(input: Tensor) {
  //       const x1 = this.conv1.forward(input)
  //       const x2 = this.conv2.forward(x1)
  //       return this.clf.forward(relu(flatten(x2)))
  //     }
  //   }
  //   const model = new MyModel()
  //   const x = random([ 50, 50, 3 ])
  //   const y = new Tensor([ 1 ])
  //   const loss = new CrossEntropyLoss()
  //   const sgd = new SGD(model.parameters(), 1e-3)
  //   const losses: any[] = []
  //   for (const e of range(100)) {
  //     sgd.resetGrad()
  //     const z = await model.forward(x)
  //     console.log(await z.raw)

  //     const l = await loss.forward(z, y)
  //     await l.backward()
  //     losses.push({ epoch: e, loss: await l.sum(), from: 'cross-entropy' })
  //     await sgd.step()
  //   }
  //   console.table(losses)
  // })
})
