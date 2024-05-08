import { flattenDeep } from 'lodash-es'
import { reshape } from 'mathjs'
import { NdArray } from 'ndarray'

export function assert(condition, msg) {
  if (!condition)
    throw new Error(`[mua] ${msg}`)
}

export class Graph<T extends object = any> {
  edges = new WeakMap<T, T[]>()
  indegree = new WeakMap<T, number>()
  nodes = new Set<T>()

  addEdge(from: T, to: T) {
    this.nodes.add(from)
    this.nodes.add(to)
    if (!this.edges.has(from))
      this.edges.set(from, [])
    this.edges.get(from)!.push(to)
    this.indegree.set(to, (this.indegree.get(to) || 0) + 1)
  }

  *sort() {
    const queue: T[] = []
    for (const node of this.nodes) {
      const indegree = this.indegree.get(node) || 0
      if (indegree === 0)
        queue.push(node)
    }
    while (queue.length) {
      const node = queue.shift()!
      yield node
      const targets = this.edges.get(node) || []
      for (const tgt of targets) {
        const indegree = this.indegree.get(tgt)! - 1
        this.indegree.set(tgt, indegree)
        if (indegree === 0)
          queue.push(tgt)
      }
    }
  }
}

export function getArrayShape(array: ArrayLike<any>) {
  const shape: number[] = []
  while (Array.isArray(array)) {
    shape.push(array.length)
    array = array[0]
  }
  return shape
}

export function toNdArray(array: ArrayLike<any>) {
  const shape = getArrayShape(array)
  const result = new Float32Array(flattenDeep(array))
  return NdArray.from(result, new Uint32Array(shape))
}

export function reshapeArray(array: ArrayLike<any>, shape: ArrayLike<number>) {
  const result = flattenDeep(array)
  return reshape(result, Array.from(shape))
}
