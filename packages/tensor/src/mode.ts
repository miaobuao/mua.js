// const config = {
//   LAZY_MODE: true,
//   REQUIRES_GRAD: true,
// }

// export function setRequiresGrad(requiresGrad: boolean) {
//   config.REQUIRES_GRAD = requiresGrad
// }

// export function getRequiresGrad() {
//   return config.REQUIRES_GRAD
// }

const singleton = {
  LAZY_MODE: true,
  REQUIRES_GRAD: true,
}

export function getConfig() {
  return singleton
}

export function setLazyMode(mode: boolean) {
  singleton.LAZY_MODE = mode
}

export function isLazyMode() {
  return singleton.LAZY_MODE
}
