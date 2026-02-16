import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path-browserify'
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  root: './frontend', // Set the root directory to frontend
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './frontend/src'),
      '@components': path.resolve(__dirname, './frontend/src/components'),
      '@pages': path.resolve(__dirname, './frontend/src/pages')
    },
    extensions: ['.js', '.ts', '.jsx', '.tsx', '.json'],
  },
  css: {
    // Configure PostCSS and ensure Tailwind is processed correctly
    postcss: {
      // Already configured in postcss.config.js
    }
  },
  build: {
    outDir: '../dist', // Output to dist directory in the project root
    sourcemap: true,
  },
  server: {
    port: 3000,
    proxy: {
      // Proxy API requests to your backend server
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      }
    }
  }
})