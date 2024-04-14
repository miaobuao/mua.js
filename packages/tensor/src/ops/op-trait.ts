import type { MaybePromise, Tensor } from '..'
import type { NdArray } from 'async-math'

export abstract class OpTrait<
    TComputeParams extends Array<unknown> = unknown[],
    TComputeReturn extends NdArray | number = any,
> {
  abstract compute(...args: TComputeParams): Promise<TComputeReturn>
  abstract gradient(grad: MaybePromise<Tensor>, ...inputs: MaybePromise<Tensor>[]): Promise<Tensor[]>
}
