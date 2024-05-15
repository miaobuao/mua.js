import { ImageHandle } from '@mua/loader'
import { tensor } from 'muajs'
import { NdArray, dtype } from 'ndarray-js'
import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'

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
    const img = ImageHandle.from(f)
    const t = new tensor.Tensor(
      new NdArray(
        img.luma16,
        {
          dtype: dtype.float32,
          shape: [ img.height, img.width, 1 ],
        },
      ),
    )
    img.free()
    resolve(t)
  })
}
