import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // During development, proxy API calls to the Flask backend.
    // In production (Option A), Flask serves everything from the same origin
    // so these paths resolve naturally — no proxy needed.
    proxy: {
      '/transcribe':  'http://localhost:5000',
      '/health':      'http://localhost:5000',
      '/model-info':  'http://localhost:5000',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
