import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/query': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/sources': 'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
    },
  },
})
