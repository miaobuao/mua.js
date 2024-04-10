export type MaybePromise<T> = T | Promise<T>

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
