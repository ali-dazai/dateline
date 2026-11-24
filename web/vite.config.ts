import { fileURLToPath } from 'node:url'
import { defineConfig, searchForWorkspaceRoot } from 'vite'
import react from '@vitejs/plugin-react'

// The API (api/server.py) runs on port 8000; /api calls are proxied there in dev.
// The preset questions live in data/presets.json beside the corpus, outside web/; the dev server may read
// that one file (the build bundles it).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
    fs: { allow: [searchForWorkspaceRoot(process.cwd()), fileURLToPath(new URL('../data/presets.json', import.meta.url))] },
  },
})
