/* eslint-disable no-console */
import type { DatasetItem } from './utils'

import { shuffle } from 'lodash-es'
import { nn, tensor } from 'muajs'
import path from 'node:path'

import { loadDataset, readImage } from './utils'
import { CrossEntropyLoss } from '../../packages/nn/src'

class MyModel extends nn.Module {
  // input: [28, 28, 4]
  readonly conv1 = new nn.Conv2d(4, 1, 3, { stride: 2 }) // output: [13, 13, 1]
  readonly conv2 = new nn.Conv2d(1, 16, 3, { stride: 2 }) // output: [6, 6, 16]
  readonly conv3 = new nn.Conv2d(16, 32, 3, { stride: 2 }) // output: [2, 2, 32]
  readonly linear1 = new nn.Linear(32 * 2 * 2, 10)
  async forward(input: tensor.Tensor) {
    let x = this.conv1.forward(input)
    x = this.conv2.forward(x)
    x = this.conv3.forward(x)
    x = tensor.flatten(x)
    x = this.linear1.forward(x)
    return tensor.relu(x, 1e-3)
  }
}

async function train(model: MyModel, ds: DatasetItem[]) {
  const optim = new nn.SGD(model.parameters(), 1e-3)
  const loss = new CrossEntropyLoss()
  const DATASIZE = 2000
  const sampled = await Promise.all(
    shuffle(ds).slice(0, DATASIZE)
      .map(async ({ path, label }) => ({ data: await readImage(path), label })),
  )
  console.log('data loaded')
  const evaluate = shuffle(ds).slice(0, 100).map(({ label, path }) => ({ label, data: readImage(path) }))
  let e = 0
  while (1) {
    e++
    const st = Date.now()
    const losses: number[] = []
    for (const { label, data } of sampled) {
      optim.resetGrad()
      const x = await data.detach()
      const z = model.forward(x)
      const l = await loss.forward(z, new tensor.Tensor([ label ]))
      await l.backward()
      await optim.step()
      losses.push(await l.sum())
    }
    const ed = Date.now()
    console.log(`[${e}] ${(ed - st) / 1000} loss: ${losses.reduce((a, b) => a + b) / DATASIZE}`)

    // if (e % 10 === 0) {
    // evaluate
    const res = await Promise.all(evaluate.map(async ({ data, label }) => {
      const x = data
      const pred = await model.forward(await x).then(d => d.raw)
      return [
        label,
        argmax(pred?.value[0]),
      ]
    }))
    const acc = res.map(([ a, b ]) => a === b ? 1 : 0).reduce((a, b) => a as any + b) / res.length
    console.log(`[${e}] evaluate: ${acc}`)
    // }
  }
}

function argmax(seq: number[]) {
  let res = 0
  let max = seq[0]
  for (let i = 1; i < seq.length; i++) {
    if (seq[i] > max) {
      max = seq[i]
      res = i
    }
  }
  return res
}

const ds = loadDataset(path.join(__dirname, './mnist'))
const model = new MyModel()
train(model, ds)
