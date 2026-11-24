import { ApiError } from './api-errors.js';

function unpack(envelope) {
  if (!envelope || typeof envelope.success !== 'boolean' || !Object.hasOwn(envelope, 'data')
      || !Object.hasOwn(envelope, 'error')) {
    throw new ApiError('流式响应格式无效，请确认 API 已更新。', 'invalid_response');
  }
  if (!envelope.success) {
    if (envelope.data !== null || typeof envelope.error?.code !== 'string'
        || typeof envelope.error?.message !== 'string') {
      throw new ApiError('流式错误响应格式无效。', 'invalid_response');
    }
    const error = envelope.error;
    throw new ApiError(error.message, error.code, error.document_id, error.details);
  }
  if (envelope.error !== null || envelope.data === null || typeof envelope.data !== 'object') {
    throw new ApiError('流式响应缺少有效数据。', 'invalid_response');
  }
  return envelope.data;
}

export async function streamAnswer({ question, model, history }, { onEvent = () => {}, signal, timeout } = {}) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) abort();
  let timedOut = false, reader;
  const abortOnTimeout = () => { timedOut = true; controller.abort(); };
  // Explicit deadlines include queue time. The normal stream budget starts at admission.
  let timer = timeout === undefined ? undefined : setTimeout(abortOnTimeout, timeout);
  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ question, model, ...(history?.length ? { history } : {}) }), signal: controller.signal,
    });
    if (timeout === undefined) timer = setTimeout(abortOnTimeout, 10 * 60 * 1000);
    if (!response.ok) {
      let envelope;
      try { envelope = await response.json(); } catch {
        throw new ApiError('服务没有返回有效的错误信息，请检查 API 连接。', 'invalid_response');
      }
      unpack(envelope);
      throw new ApiError('流式请求失败（HTTP ' + response.status + '）。', 'invalid_response');
    }
    if (!response.headers.get('content-type')?.includes('text/event-stream') || !response.body) {
      throw new ApiError('服务不支持当前流式接口，请重启更新后的 API。', 'invalid_response');
    }
    reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8', { fatal: true });
    let buffer = '', eventName = '', dataLines = [], eventSize = 0;
    let final = null;
    function dispatch() {
      if (!dataLines.length) { eventName = ''; return; }
      let envelope;
      try { envelope = JSON.parse(dataLines.join('\n')); } catch {
        throw new ApiError('流式响应包含无效 JSON。', 'invalid_response');
      }
      const data = unpack(envelope);
      const type = eventName || 'message';
      if (type === 'delta' && typeof data.text !== 'string'
          || type === 'stage' && !['retrieving', 'generating', 'retrying', 'verifying'].includes(data.stage)
          || !['delta', 'stage', 'reset', 'result'].includes(type)) {
        throw new ApiError('流式事件格式无效。', 'invalid_response');
      }
      if (type === 'result') {
        if (typeof data.answer !== 'string' || !['answered', 'insufficient_evidence'].includes(data.status)) {
          throw new ApiError('流式回答没有有效的最终结果。', 'invalid_response');
        }
        final = data;
      }
      onEvent({ type, data });
      eventName = ''; dataLines = []; eventSize = 0;
    }
    while (true) {
      const { value, done } = await reader.read();
      if (controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
      try { buffer += decoder.decode(value, { stream: !done }); } catch {
        throw new ApiError('流式回答字符编码无效。', 'invalid_response');
      }
      if (buffer.length + eventSize > 1024 * 1024) {
        throw new ApiError('流式事件超过允许大小。', 'invalid_response');
      }
      let end;
      while ((end = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, end).replace(/\r$/, '');
        buffer = buffer.slice(end + 1);
        if (!line) dispatch();
        else if (!line.startsWith(':')) {
          const colon = line.indexOf(':');
          const field = colon === -1 ? line : line.slice(0, colon);
          const value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, '');
          if (field === 'event') eventName = value;
          if (field === 'data') { dataLines.push(value); eventSize += value.length; }
        }
        if (final) return final;
      }
      if (done) throw new ApiError('连接已结束，但未收到完整回答。请重试。', 'stream_incomplete');
    }
  } catch (error) {
    if (controller.signal.aborted || error.name === 'AbortError') {
      throw new ApiError(timedOut ? '等待回答超时，请稍后重试。' : '已停止本次回答。',
        timedOut ? 'request_timeout' : 'request_cancelled');
    }
    if (error instanceof ApiError) throw error;
    throw new ApiError('流式连接中断，请确认本地服务正常运行后重试。', 'connection_error');
  } finally {
    if (timer !== undefined) clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
    if (reader) {
      try { await reader.cancel(); } catch { /* Connection may already be closed. */ }
      reader.releaseLock();
    }
    controller.abort();
  }
}
