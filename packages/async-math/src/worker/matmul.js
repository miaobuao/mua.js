import { multiply as mul } from 'mathjs'
import { Worker, isMainThread, parentPort } from 'node:worker_threads'

import { WorkerPool } from './pool'

/** @type {WorkerPool} */
let workers

if (isMainThread)
  workers = new WorkerPool(() => new Worker(__filename), 3)

export async function multiply(a, b) {
  return new Promise((resolve, reject) => {
    workers.deliver([ a, b ], resolve)
  })
}

if (!isMainThread) {
  parentPort.on('message', ({ taskId, args }) => {
    parentPort.postMessage({
      taskId,
      data: mul(...args),
    })
  })
}
