import type { NdArrayNumberCell } from './ndarray'

import { map, zeros } from 'mathjs'

export function normal(size: number[], mean = 0, std = 0.01) {
  return map(zeros(size as number[]), () => normalRandom(mean, std)) as NdArrayNumberCell[]
}

function normalRandom(mean: number, std: number) {
  let u = 0.0
  let v = 0.0
  let w = 0.0
  let c = 0.0
  do {
    u = Math.random() * 2 - 1.0
    v = Math.random() * 2 - 1.0
    w = u * u + v * v
  } while (w === 0.0 || w >= 1.0)
  // Box-Muller
  c = Math.sqrt((-2 * Math.log(w)) / w)
  const normal = mean + (u * c) * std
  return normal
}
