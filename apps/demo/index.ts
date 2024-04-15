/* eslint-disable no-console */
import type { DatasetItem } from './utils'

import { shuffle } from 'lodash-es'
import { nn, tensor } from 'muajs'
import path from 'node:path'

import { loadDataset, readImage } from './utils'
import { CrossEntropyLoss } from '../../packages/nn/src'

class MyModel extends nn.Module {
  readonly conv1 = new nn.Conv2d(4, 1, 3, { stride: 2 })
  readonly linear1 = new nn.Linear(169, 10)
  forward(input: tensor.Tensor) {
    let x = this.conv1.forward(input)
    x = tensor.flatten(x)
    return this.linear1.forward(tensor.relu(x, 1e-3))
  }
}

async function train(model: MyModel, ds: DatasetItem[]) {
  const optim = new nn.SGD(model.parameters(), 1e-3)
  const loss = new CrossEntropyLoss()
  const DATASIZE = 5000
  const sampled = shuffle(ds).slice(0, DATASIZE)
  let e = 0
  while (1) {
    e++
    const st = Date.now()
    const losses: number[] = []
    for (const { label, path } of sampled) {
      optim.resetGrad()
      const x = await readImage(path)
      const z = model.forward(x)
      const l = await loss.forward(z, new tensor.Tensor([ label ]))
      await l.backward()
      await optim.step()
      losses.push(await l.sum())
    }
    const ed = Date.now()
    console.log(`[${e}] ${(ed - st) / 1000} loss: ${losses.reduce((a, b) => a + b) / DATASIZE}`)
  }
}

const ds = loadDataset(path.join(__dirname, './mnist'))
const model = new MyModel()
train(model, ds)
