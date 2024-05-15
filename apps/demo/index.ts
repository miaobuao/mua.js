import type { DatasetItem } from './utils'

import consola from 'consola'
import { mean, shuffle } from 'lodash-es'
import { nn, tensor } from 'muajs'
import path from 'node:path'

import { loadDataset, readImage } from './utils'
/** CNN */
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

/** MLP */
// class MyModel extends nn.Module {
//   // input: [28, 28, 1]
//   readonly linear1 = new nn.Linear(28 * 28, 32)
//   readonly linear2 = new nn.Linear(32, 10)

//   async forward(x: MaybePromise<tensor.Tensor>) {
//     x = tensor.flatten(x)
//     x = this.linear1.forward(x)
//     x = tensor.tanh(x)
//     x = this.linear2.forward(x)
//     return x
//   }
// }

async function train(model: MyModel, ds: DatasetItem[]) {
  const optim = new nn.SGD(model.parameters(), 1e-3)
  const loss = new nn.CrossEntropyLoss()
  const DATASIZE = 3000
  const sampled = shuffle(ds).slice(0, DATASIZE)
  const evaluate = shuffle(ds).slice(0, 100)
  const forwardTime: number[] = []
  const backwardTime: number[] = []
  let e = 0
  while (1) {
    const losses: number[] = []
    for (const { label, path } of sampled) {
      ++e
      optim.resetGrad()
      const data = await readImage(path)
      const x = await data.detach()
      const y = new tensor.Tensor([ label ], { requiresGrad: false })

      let st = Date.now()
      const z = await model.forward(x)
      const l = await loss.forward(z, y)
      let ed = Date.now()
      forwardTime.push(ed - st)

      st = Date.now()
      await l.backward()
      await optim.step()
      ed = Date.now()
      backwardTime.push(ed - st)

      const lossVal = await l.sum()
      losses.push(lossVal)
      if (e % 500 === 0) {
        // evaluate
        const res = await Promise.all(evaluate.map(async ({ path, label }) => {
          const x = await readImage(path)
          x.requiresGrad = false
          const pred = await model.forward(x).then(d => d.raw)
          return [
            label,
            await pred?.argmax(),
          ]
        }))

        const acc = res.map(([ a, b ]) => a === b ? 1 : 0).reduce((a, b) => a as any + b) / res.length
        consola.log('evaluation', {
          epoch: e,
          acc,
        })
      }
    }
    const backMeanTime = mean(backwardTime)
    const forwardMeanTime = mean(forwardTime)
    consola.log({
      epoch: e,
      forward: forwardMeanTime,
      backward: backMeanTime,
      loss: losses.reduce((a, b) => a + b) / DATASIZE,
    })
  }
}

const ds = loadDataset(path.join(__dirname, './MNIST'))
const model = new MyModel()
train(model, ds)
