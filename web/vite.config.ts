import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  
  // Development server configuration
  server: {
    port: 3000,
    host: true, // Listen on all addresses
    cors: true,
    proxy: {
      // Proxy API requests to backend
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      // WebSocket proxy for real-time updates
      '/socket.io': {
        target: process.env.VITE_WS_URL || 'http://localhost:8080',
        ws: true,
      },
    },
  },
  
  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunk for external libraries
          vendor: ['react', 'react-dom', 'react-router-dom'],
          // D3 chunk for visualization libraries
          d3: ['d3', 'd3-selection', 'd3-scale', 'd3-axis', 'd3-shape', 'd3-time', 'd3-time-format', 'd3-zoom', 'd3-drag'],
          // UI chunk for UI components
          ui: ['antd', '@ant-design/icons'],
        },
      },
    },
    // Increase chunk size warning limit for D3 bundles
    chunkSizeWarningLimit: 1000,
  },
  
  // Path resolution
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@styles': resolve(__dirname, 'src/styles'),
      '@types': resolve(__dirname, 'src/types'),
      '@hooks': resolve(__dirname, 'src/hooks'),
      '@store': resolve(__dirname, 'src/store'),
      '@services': resolve(__dirname, 'src/services'),
    },
  },
  
  // Environment variables
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
  },
  
  // CSS configuration
  css: {
    preprocessorOptions: {
      scss: {
        additionalData: `@import "@/styles/variables.scss";`,
      },
    },
  },
  
  // Test configuration
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
      ],
    },
  },
  
  // Optimization
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'd3',
      'antd',
      'axios',
      'lodash',
      'dayjs',
    ],
  },
})