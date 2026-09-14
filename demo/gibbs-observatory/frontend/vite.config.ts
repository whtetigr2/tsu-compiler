import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Launcher sets VITE_API_PROXY_TARGET / GIBBS_API_PORT so we never silently
// talk to another demo on :8000 (e.g. lithography / ThermoLith stacks).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const api =
    env.VITE_API_PROXY_TARGET ||
    (env.GIBBS_API_PORT
      ? `http://127.0.0.1:${env.GIBBS_API_PORT}`
      : 'http://127.0.0.1:8088')

  return {
    plugins: [react()],
    server: {
      port: 5188,
      strictPort: false,
      proxy: {
        '/api': api,
        '/ws': {
          target: api.replace(/^http/, 'ws'),
          ws: true,
        },
      },
    },
  }
})
