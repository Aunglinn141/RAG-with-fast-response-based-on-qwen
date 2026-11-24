import assert from 'node:assert/strict';
import test, { before, after } from 'node:test';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';
import react from '@vitejs/plugin-react';

let server, Workspace, App, StreamAnswer, Library, UploadQueue, AnswerContent;
before(async () => {
  const root = fileURLToPath(new URL('..', import.meta.url));
  server = await createServer({
    root, configFile: false, plugins: [react()], logLevel: 'error',
    resolve: { preserveSymlinks: true },
    ssr: { external: ['react', 'react-dom', 'react-dom/server'] },
    define: { __LOCAL_RAG_API_URL__: JSON.stringify('http://127.0.0.1:8000') },
    cacheDir: resolve(root, '..', '.runtime', 'vite-test-cache'),
    server: { middlewareMode: true, hmr: false, watch: null },
    optimizeDeps: { noDiscovery: true, include: [] },
    appType: 'custom',
  });
  Workspace = (await server.ssrLoadModule('/src/components/SearchWorkspace.jsx')).default;
  App = (await server.ssrLoadModule('/src/App.jsx')).default;
  StreamAnswer = (await server.ssrLoadModule('/src/components/StreamingAnswer.jsx')).default;
  Library = (await server.ssrLoadModule('/src/components/DocumentLibrary.jsx')).default;
  UploadQueue = (await server.ssrLoadModule('/src/components/UploadQueue.jsx')).default;
  AnswerContent = (await server.ssrLoadModule('/src/components/AnswerContent.jsx')).default;
});
after(async () => { await server?.close(); });

const documents = [{
  document_id: 'private-document-identifier', filename: '内部客户名单.pdf',
  status: 'indexed', chunk_count: 173, generation: 'private-generation', model_version: 'private-digest',
}];
const props = {
  documents, busy: false, revision: 0, onSearchBusy() {}, onDetail() {}, onLibrary() {}, onError() {},
};

test('question view renders no document names, selection, counts, source mode or management cards', () => {
  const html = renderToStaticMarkup(React.createElement(Workspace, { ...props, mode: 'ask' }));
  for (const secret of ['内部客户名单.pdf', 'private-document-identifier', 'private-generation', 'private-digest',
    '检索范围', '检索数量', '全部已索引文档', '173', '份文档可供检索', '个文本块已建立索引', '管理文档库', '原文检索']) {
    assert.equal(html.includes(secret), false, secret);
  }
  assert.match(html, /有什么想了解的/);
  assert.match(html, /回答模型/);
  assert.match(html, /提问/);
});

test('separate source-search view retains document debugging controls', () => {
  const html = renderToStaticMarkup(React.createElement(Workspace, { ...props, mode: 'search' }));
  assert.match(html, /内部客户名单.pdf/);
  assert.match(html, /检索范围/);
  assert.match(html, /检索数量/);
  assert.match(html, /原文检索/);
});

test('application opens directly on Q&A without rendering a hidden document library', () => {
  const html = renderToStaticMarkup(React.createElement(App));
  assert.match(html, /有什么想了解的/);
  for (const text of ['aria-label="文档列表"', '选择文件', '所有文档', '上传一份文档', 'API 文档', 'nav-count']) {
    assert.equal(html.includes(text), false, text);
  }
  assert.match(html, /文档管理/);
  for (const text of ['文物档案', '建立档案', '文物基础问答', '导入 JSON']) {
    assert.equal(html.includes(text), false, text);
  }
});

test('streaming view shows live answer text as an incomplete draft without source cards', () => {
  const html = renderToStaticMarkup(React.createElement(StreamAnswer, {
    text: '逐步显示的中文回答<script>bad</script>', stage: 'generating', elapsed: 4,
  }));
  assert.match(html, /逐步显示的中文回答/);
  assert.match(html, /内容尚未完整/);
  assert.match(html, /aria-busy="true"/);
  assert.match(html, /&lt;script&gt;/);
  assert.equal(html.includes('<script>bad</script>'), false);
  for (const hidden of ['source-card', '检索范围', 'document_id', 'evidence']) assert.equal(html.includes(hidden), false);
});

test('empty stream shows model loading state and finalization stage is explicit', () => {
  const waiting = renderToStaticMarkup(React.createElement(StreamAnswer, { text: '', stage: 'waiting', elapsed: 30 }));
  assert.match(waiting, /等待服务响应/);
  assert.match(waiting, /排队/);
  const loading = renderToStaticMarkup(React.createElement(StreamAnswer, { text: '', stage: 'retrieving', elapsed: 0 }));
  assert.match(loading, /正在查找相关信息/);
  const verifying = renderToStaticMarkup(React.createElement(StreamAnswer, { text: '答案', stage: 'verifying', elapsed: 3 }));
  assert.match(verifying, /正在整理回答/);
});

test('document management supports multiple files and queue controls', () => {
  const html = renderToStaticMarkup(React.createElement(Library, { documents: [], onUpload() {} }));
  assert.match(html, /type="file" multiple=""/);
  assert.match(html, /批量导入/);
  assert.match(html, /选择文件（可多选）/);
  const queue = renderToStaticMarkup(React.createElement(UploadQueue, { active: false, items: [
    { id: '1', filename: '<bad>.pdf', status: 'failed', message: '解析失败' },
    { id: '2', filename: 'pending.md', status: 'paused' },
  ] }));
  assert.match(queue, /重试失败项/);
  assert.match(queue, /继续未导入项/);
  assert.match(queue, /&lt;bad&gt;/);
});

test('conversation layout displays answer text without retrieved sources', () => {
  const html = renderToStaticMarkup(React.createElement(Workspace, { ...props, mode: 'ask',
    conversation: { id: 'c1', messages: [{ id: 'm1', question: '测试问题', answer: '**回答**', status: 'answered', model: 'qwen3:1.7b', elapsed: 2,
      sources: [{ text: 'DO-NOT-RENDER-PRIVATE-SOURCE' }] }] },
  }));
  assert.match(html, /chat-user/);
  assert.match(html, /chat-composer/);
  assert.match(html, /<strong>回答<\/strong>/);
  assert.match(html, /复制回答/);
  assert.equal(html.includes('DO-NOT-RENDER-PRIVATE-SOURCE'), false);
});

test('answer formatting renders code and lists while escaping model HTML', () => {
  const html = renderToStaticMarkup(React.createElement(AnswerContent, { text: '# 标题\n\n1. **重点**\n2. 第二项\n\n<script>alert(1)</script>\n\n```python\nprint("hello")\n```' }));
  assert.match(html, /<ol>/);
  assert.match(html, /<pre><code>/);
  assert.match(html, /&lt;script&gt;/);
  assert.equal(html.includes('<script>'), false);
});
