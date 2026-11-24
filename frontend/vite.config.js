import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const target = process.env.LOCAL_RAG_API_URL || 'http://127.0.0.1:8000';
const url = new URL(target);
if (url.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)
    || url.username || url.password || url.pathname !== '/' || url.search || url.hash) {
  throw new Error('LOCAL_RAG_API_URL must be a local HTTP origin, for example http://127.0.0.1:8000');
}

function apiProxy() {
  const idleTimeout = 30 * 60 * 1000;
  return {
    '^/api(?:/|$)': {
      target,
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api(?=\/|$)/, '') || '/',
      configure(proxy) {
        proxy.on('proxyReq', (upstream, request, response) => {
          const watchTransfer = () => {
            response.setTimeout(idleTimeout);
            upstream.setTimeout(idleTimeout, () => upstream.destroy());
          };
          // The API queues its POST, PUT and DELETE operations before sending headers.
          if (['POST', 'PUT', 'DELETE'].includes(request.method)) {
            response.setTimeout(0);
            upstream.once('response', watchTransfer);
          } else {
            watchTransfer();
          }
        });
        proxy.on('error', (_error, _request, response) => {
          if (response && typeof response.writeHead === 'function' && !response.headersSent) {
            response.writeHead(503, { 'Content-Type': 'application/json; charset=utf-8' });
            response.end(JSON.stringify({ success: false, data: null, error: {
              code: 'backend_unavailable',
              message: '无法连接本地 API，请运行 scripts/start_api.ps1 并保持服务运行。',
            } }));
          }
        });
      },
    },
  };
}

export default defineConfig({
  define: { __LOCAL_RAG_API_URL__: JSON.stringify(url.origin) },
  plugins: [react()],
  // Keep dependency paths stable in local Windows checkouts, including restricted shells.
  resolve: { preserveSymlinks: true },
  server: { host: '127.0.0.1', port: 5173, strictPort: true, proxy: apiProxy() },
  preview: { host: '127.0.0.1', port: 5173, strictPort: true, proxy: apiProxy() },
});
