import { tensor } from 'muajs'
import { createReadStream, readdirSync } from 'node:fs'
import path from 'node:path'
import { PNG } from 'pngjs'

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
    createReadStream(path).pipe(new PNG()).on('parsed', function () {
      resolve(new tensor.Tensor(Array.from(this.data)).reshape([ 28, 28, 4 ]))
    })
    // getPixels(path, (err, pixels) => {
    //   if (err)
    //     reject(err)
    //   else
    //     resolve(new tensor.Tensor(Array.from(pixels.data)).reshape([ 28, 28, 4 ]))
    // })
  })
}

export interface DatasetItem {
  path: string
  label: number
}
