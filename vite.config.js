import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,       // Desired port
    strictPort: true, // Fail if port is in use instead of switching
    host: true,       // If you want to bind 0.0.0.0 (optional)
  },
})
