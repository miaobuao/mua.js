/* eslint-disable no-console */
import type { DatasetItem } from './utils'

import { shuffle } from 'lodash-es'
import { nn, tensor } from 'muajs'
import path from 'node:path'

import { loadDataset, readImage } from './utils'
// import { CrossEntropyLoss } from '../../packages/nn/src'

class MyModel extends nn.Module {
  // input: [28, 28, 1]
  readonly conv1 = new nn.Conv2d(1, 32, 5, { stride: 2 }) // output: [12, 12, 32],
  readonly conv2 = new nn.Conv2d(32, 2, 5, { stride: 2 }) // ouput: [4, 4, 2]
  readonly linear1 = new nn.Linear(32, 10)

  async forward(input: tensor.Tensor) {
    let x = this.conv1.forward(input)
    x = tensor.tanh(x)
    x = this.conv2.forward(x)
    x = tensor.flatten(x)
    x = this.linear1.forward(x)
    x = tensor.relu(x, 1e-3)
    return x
  }
}

async function train(model: MyModel, ds: DatasetItem[]) {
  const optim = new nn.SGD(model.parameters(), 1e-3)
  const loss = new nn.CrossEntropyLoss()
  const DATASIZE = 2000
  const sampled = shuffle(ds).slice(0, DATASIZE)
  console.log('data loaded')
  const evaluate = shuffle(ds).slice(0, 100)
  let e = 0
  while (1) {
    e++
    const st = Date.now()
    const losses: number[] = []
    for (const { label, path } of sampled) {
      optim.resetGrad()
      const data = await readImage(path)
      const x = await data.detach()
      const z = model.forward(x)
      const l = await loss.forward(z, new tensor.Tensor([ label ]))
      await l.backward()
      await optim.step()
      const lossVal = await l.sum()
      losses.push(lossVal)
    }
    const ed = Date.now()
    console.log(`[${e}] ${(ed - st) / 1000} loss: ${losses.reduce((a, b) => a + b) / DATASIZE}`)

    const res = await Promise.all(evaluate.map(async ({ path, label }) => {
      const x = await readImage(path)
      const pred = await model.forward(x).then(d => d.raw)
      return [
        label,
        pred?.argmax(),
      ]
    }))
    const acc = res.map(([ a, b ]) => a === b ? 1 : 0).reduce((a, b) => a as any + b) / res.length
    console.log(`[${e}] evaluate: ${acc}`)
  }
}

const ds = loadDataset(path.join(__dirname, './MNIST'))
const model = new MyModel()
train(model, ds)
