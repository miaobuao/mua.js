import type { Tensor } from '..'
import type { MaybePromise } from '@mua/common'
import type { NdArray } from 'async-math'

export abstract class OpTrait<
    TComputeParams extends Array<unknown> = unknown[],
> {
  abstract compute(...args: TComputeParams): Promise<NdArray | [number]>
  abstract gradient(grad: MaybePromise<Tensor>, ...inputs: MaybePromise<Tensor>[]): Promise<Tensor[]>
}
