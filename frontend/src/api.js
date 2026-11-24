const ROOT = '/api';

import { ApiError } from './api-errors.js';
import { streamAnswer } from './streaming.js';
export { ApiError } from './api-errors.js';

async function request(path, { timeout = 15000, bodyTimeout, signal, ...options } = {}) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) abort();
  // Queued JSON operations return their headers only after execution completes.
  let timer = bodyTimeout === undefined ? setTimeout(abort, timeout) : undefined;
  try {
    const response = await fetch(ROOT + path, { ...options, signal: controller.signal });
    if (bodyTimeout !== undefined) timer = setTimeout(abort, bodyTimeout);
    const envelope = await response.json().catch(error => {
      if (error.name === 'AbortError') throw error;
      return null;
    });
    const validEnvelope = envelope && typeof envelope === 'object' && !Array.isArray(envelope)
      && typeof envelope.success === 'boolean' && Object.hasOwn(envelope, 'data') && Object.hasOwn(envelope, 'error');
    const validError = envelope?.error && typeof envelope.error === 'object' && !Array.isArray(envelope.error)
      && typeof envelope.error.code === 'string' && envelope.error.code.trim()
      && typeof envelope.error.message === 'string' && envelope.error.message.trim();
    if (!validEnvelope || (envelope.success
      ? envelope.data === null || envelope.error !== null || !response.ok
      : envelope.data !== null || !validError)) {
      throw new ApiError(`服务返回格式不符合当前 API 约定（HTTP ${response.status}）。请确认已重启更新后的 API。`, 'invalid_response');
    }
    if (!envelope.success) {
      const error = envelope.error;
      const details = Array.isArray(error.details) ? error.details : [];
      const validation = details.map(item => typeof item?.msg === 'string' ? item.msg : '').filter(Boolean).join('；');
      throw new ApiError(validation ? `${error.message}：${validation}` : error.message,
        error.code, error.document_id, details);
    }
    return envelope.data;
  } catch (error) {
    if (signal?.aborted) throw new ApiError('已停止本次请求。', 'request_cancelled');
    if (error instanceof ApiError) throw error;
    if (error.name === 'AbortError') {
      throw new ApiError('等待响应超时。服务可能仍在处理，请刷新文档状态后再决定是否重试。', 'request_timeout');
    }
    throw new ApiError('无法连接到文档服务，请确认本地 API 已在 8000 端口启动。', 'connection_error');
  } finally {
    if (timer !== undefined) clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}

const longRequest = 30 * 60 * 1000;
export const api = {
  askStream: streamAnswer,
  health: () => request('/health', { timeout: 8000 }),
  models: () => request('/models', { timeout: 60000 }),
  documents: () => request('/documents', { timeout: 35000 }),
  document: id => request(`/documents/${encodeURIComponent(id)}`),
  upload: file => {
    const body = new FormData();
    body.append('file', file);
    return request('/documents', { method: 'POST', body, bodyTimeout: longRequest });
  },
  reindex: id => request(`/documents/${encodeURIComponent(id)}/reindex`, { method: 'POST', bodyTimeout: longRequest }),
  remove: id => request(`/documents/${encodeURIComponent(id)}`, { method: 'DELETE', bodyTimeout: longRequest }),
  search: (body, { signal } = {}) => request('/search', { method: 'POST', body: JSON.stringify(body), signal,
    headers: { 'Content-Type': 'application/json' }, bodyTimeout: 5 * 60 * 1000 }),
  ask: ({ question, model, history }, { signal } = {}) => request('/chat', { method: 'POST', body: JSON.stringify({ question, model, ...(history?.length ? { history } : {}) }), signal,
    headers: { 'Content-Type': 'application/json' }, bodyTimeout: 6 * 60 * 1000 }),
};

export function downloadUrl(id, kind = 'original') {
  return `${ROOT}/documents/${encodeURIComponent(id)}/${kind}`;
}
