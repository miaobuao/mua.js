import { nn, tensor } from 'muajs'
import { exit } from 'node:process'

async function main() {
  const linear = new nn.Linear(100, 500)
  const x = tensor.normal([ 200, 100 ])
  const y = tensor.normal([ 200, 500 ])
  const loss = new nn.L2Loss()
  const optim = new nn.SGD(linear.parameters(), 1e-3)
  for (let i = 0; i < 10000; i++) {
    const st = Date.now()
    optim.resetGrad()
    const z = await linear.forward(x)
    const l = await loss.forward(z, y)
    await l.backward()
    await optim.step()
    // await z.raw
    const ed = Date.now()

    // eslint-disable-next-line no-console
    console.log(i, ed - st, await l.buffer)
    if (ed - st > 100)
      exit()
  }
}

main()
