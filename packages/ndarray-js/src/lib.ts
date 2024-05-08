export function getArrayShape(array: ArrayLike<any>) {
  const shape: number[] = []
  while (Array.isArray(array)) {
    shape.push(array.length)
    array = array[0]
  }
  return shape
}

export function getStrides(shape: number[]) {
  return shape.map((_, i) => shape.slice(i + 1, shape.length).reduce((a, b) => a * b, 1))
}
