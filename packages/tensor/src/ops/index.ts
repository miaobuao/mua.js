import type { Tensor } from '..'
import type { MathCollection } from 'async-math'

export abstract class OpTrait<
    TComputeParams extends Array<unknown> = unknown[],
    TComputeReturn extends MathCollection | number = any,
    TGradientReturn = any,
> {
  abstract compute(...args: TComputeParams): Promise<TComputeReturn>
  abstract gradient(grad: Tensor, inputs: Tensor[]): Promise<TGradientReturn>
}

export * from './add'
export * from './add-scalar'
export * from './dot'
export * from './matmul'
