export * from './conv'

export type MaybePromise<T> = Promise<T> | T

export async function asyncValueNotNil<T = unknown>(value: MaybePromise<T>, e?: Error): Promise<NonNullable<T>> {
  const v = await value
  if (v === undefined || v === null)
    throw e || new Error('this field should not be null')
  return v
}
