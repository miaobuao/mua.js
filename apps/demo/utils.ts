import { tensor } from 'muajs'
import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'

import { loadImageByLuma } from '../../packages/tensor/src'

export interface DatasetItem {
  path: string
  label: number
}

export function loadDataset(dir: string) {
  const root = path.resolve(dir)
  return readdirSync(root).map((label) => {
    return readdirSync(path.join(root, label)).map((name) => {
      return {
        path: path.join(root, label, name),
        label: Number(label),
      }
    })
  }).flat()
}

export function readImage(path: string) {
  return new Promise<tensor.Tensor>((resolve, reject) => {
    const f = readFileSync(path)
    const t = new tensor.Tensor(
      loadImageByLuma(f),
    )
    resolve(t)
  })
}
