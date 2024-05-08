import { v1 } from 'uuid'

export class WorkerPool {
  /**
   *
   * @param {()=>import("worker_threads").Worker} creator
   * @param {number} count
   */
  constructor(creator, count) {
    /** @type {import("worker_threads").Worker[]} */
    this.pool = []
    this.count = 0
    this.i = 0
    this.callbackMap = new Map()
    for (let i = 0; i < count; ++i)
      this.appendWorker(creator())
  }

  /**
   *
   * @param {import("worker_threads").Worker} worker
   */
  appendWorker(worker) {
    worker.on('message', ({ taskId, data }) => {
      this.callbackMap.get(taskId)(data)
      this.callbackMap.delete(taskId)
    })
    this.pool.push(worker)
    this.count++
  }

  deliver(args, resolve) {
    this.i = (++this.i) % this.count
    const taskId = v1()
    this.pool[this.i].postMessage({ taskId, args })
    this.callbackMap.set(taskId, resolve)
  }
}
