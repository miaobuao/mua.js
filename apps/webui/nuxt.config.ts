import topLevelAwait from 'vite-plugin-top-level-await'
import wasm from 'vite-plugin-wasm'

export default defineNuxtConfig({
  devtools: { enabled: true },
  ssr: false,
  vite: {
    plugins: [
      wasm(),
      topLevelAwait(),
    ],
  },
})
