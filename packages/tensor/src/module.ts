import { forIn } from 'lodash-es'

import { Parameter } from '.'

export class Module {
  parameters() {
    return getParameters(this)
  }
}

export function getParameters(module: Module) {
  let res: Parameter[] = []
  forIn(module, (v: unknown) => {
    if (v instanceof Parameter)
      res.push(v)
    else if (v instanceof Module)
      res = [ ...res, ...getParameters(v) ]
  })
  return res
}
