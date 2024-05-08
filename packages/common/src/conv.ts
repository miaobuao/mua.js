/**
 *
 * @param size [feat_in, cin]
 * @param kernelSize [k_size, cin, cout]
 * @param stride
 * @param padding
 * @returns [feat_out, cout]
 */
export function getConv1dSize(size: ArrayLike<number>, kernelSize: ArrayLike<number>, stride = 1, padding = 0) {
  if (size.length !== 2)
    throw new Error(`conv1d: size must be 2D matrix, got ${size.length}D matrix`)

  if (kernelSize.length !== 3)
    throw new Error(`conv1d: kernel must be 3D matrix, got ${kernelSize.length}D matrix`)

  return [
    ...getConvSize([ size[0]! ], [ kernelSize[0]! ], stride, padding),
    kernelSize[2]!,
  ] as [number, number]
}

/**
 *
 * @param size [h, w, cin]
 * @param kernelSize [kh, kw, cin, cout]
 * @param stride
 * @param padding
 * @returns [hout, wout, cout]
 */
export function getConv2dSize(size: ArrayLike<number>, kernelSize: ArrayLike<number>, stride = 1, padding = 0) {
  if (size.length !== 3)
    throw new Error(`conv2d: size must be 3D matrix, got ${size.length}D matrix`)

  if (kernelSize.length !== 4)
    throw new Error(`conv2d: kernel must be 4D matrix, got ${kernelSize.length}D matrix`)

  return [
    ...getConvSize(
      Array.from(size).slice(0, 2),
      Array.from(kernelSize).slice(0, 2),
      stride,
      padding,
    ),
    kernelSize[3]!,
  ] as [number, number, number]
}

export function getConvSize(size: number[], kernelSize: number[], stride = 1, padding = 0) {
  const outSize: number[] = []
  for (let i = 0; i < size.length; i++)
    outSize.push(Math.floor((size[i]! + 2 * padding - kernelSize[i]!) / stride + 1))
  return outSize
}

export function getPoolSize(size: number[], kernelSize: number[], stride: number | number[] | undefined | null = undefined, padding = 0) {
  stride ??= kernelSize
  const outSize: number[] = []
  for (let i = 0; i < size.length; i++) {
    const s = Array.isArray(stride) ? stride[i]! : stride
    outSize.push(Math.floor((size[i]! + 2 * padding - kernelSize[i]!) / s + 1))
  }
  return outSize
}
