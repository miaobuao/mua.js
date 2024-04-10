import { Tensor } from '@mua/tensor'
import { forIn } from 'lodash-es'

export abstract class Module<
  TForwardParams extends unknown[] = unknown[],
  TForwardReturnValue = any,
  TForwardReturn = Promise<TForwardReturnValue>,
> {
  parameters() {
    return getParameters(this as any)
  }

  abstract forward(...params: TForwardParams): TForwardReturn
}

export function getParameters(module: Module) {
  let res: Tensor[] = []
  forIn(module, (v: unknown) => {
    if (v instanceof Tensor)
      res.push(v)
    else if (v instanceof Module)
      res = [ ...res, ...getParameters(v) ]
  })
  return res
}
