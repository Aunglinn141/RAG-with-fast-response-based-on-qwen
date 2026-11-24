import assert from 'node:assert/strict';
import test from 'node:test';
import { api, ApiError } from '../src/api.js';

const envelope = data => ({ success: true, data, error: null });
const event = (type, data) => 'event: ' + type + '\r\ndata: ' + JSON.stringify(envelope(data)) + '\r\n\r\n';
const final = { answer: '中文📚回答', status: 'answered', sources: [], evidence: [] };
function response(text, oneByte = false) {
  const bytes = new TextEncoder().encode(text);
  return new Response(new ReadableStream({
    start(controller) {
      if (oneByte) for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      else controller.enqueue(bytes);
      controller.close();
    },
  }), { headers: { 'Content-Type': 'text/event-stream; charset=utf-8' } });
}

test('stream decodes split UTF-8 and CRLF, keeps scope private, and emits before terminal result', async t => {
  const events = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, '/api/chat/stream');
    assert.deepEqual(JSON.parse(options.body), { question: '问题', model: 'qwen3:1.7b' });
    assert.equal(options.headers.Accept, 'text/event-stream');
    return response(': keepalive\r\n\r\n' + event('stage', { stage: 'generating' })
      + event('delta', { text: '中文📚' }) + event('delta', { text: '回答' }) + event('result', final), true);
  });
  const result = await api.askStream({ question: '问题', model: 'qwen3:1.7b', top_k: 4, document_id: 'private' },
    { onEvent: e => events.push(e) });
  assert.deepEqual(result, final);
  assert.deepEqual(events.map(e => e.type), ['stage', 'delta', 'delta', 'result']);
  assert.equal(events.filter(e => e.type === 'delta').map(e => e.data.text).join(''), final.answer);
});

test('delta callback runs while server has not sent result yet', async t => {
  let controller, resolveDelta, completed = false;
  const seen = new Promise(resolve => { resolveDelta = resolve; });
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({
    start(c) { controller = c; c.enqueue(new TextEncoder().encode(event('delta', { text: '正在回答' }))); },
  }), { headers: { 'Content-Type': 'text/event-stream' } }));
  const pending = api.askStream({ question: '问题' }, { onEvent: e => { if (e.type === 'delta') resolveDelta(); } })
    .then(value => { completed = true; return value; });
  await seen;
  assert.equal(completed, false);
  controller.enqueue(new TextEncoder().encode(event('result', final)));
  controller.close();
  assert.deepEqual(await pending, final);
});

test('default stream timeout begins when the queued request receives headers', async t => {
  const result = response(event('result', final));
  let duration;
  t.mock.method(globalThis, 'setTimeout', (_callback, ms) => { duration = ms; return 123; });
  t.mock.method(globalThis, 'clearTimeout', id => assert.equal(id, 123));
  t.mock.method(globalThis, 'fetch', async () => {
    assert.equal(duration, undefined);
    return result;
  });
  await api.askStream({ question: '问题' });
  assert.equal(duration, 10 * 60 * 1000);
});

test('streaming sends the current conversation history with the selected model', async t => {
  const history = [{ question: '介绍测试壶', answer: '这是一件青铜器。' }];
  t.mock.method(globalThis, 'fetch', async (_url, options) => {
    assert.deepEqual(JSON.parse(options.body), { question: '它在哪里？', model: 'qwen3:1.7b', history });
    return response(event('result', final));
  });
  await api.askStream({ question: '它在哪里？', model: 'qwen3:1.7b', history });
});

test('reset separates retries and final insufficient-evidence result is authoritative', async t => {
  const types = [];
  const rejected = { ...final, answer: '引用无法核对', status: 'insufficient_evidence', reason: 'invalid_evidence' };
  t.mock.method(globalThis, 'fetch', async () => response(event('delta', { text: '旧草稿' })
    + event('reset', { reason: 'generation_retry' }) + event('delta', { text: '新草稿' })
    + event('result', rejected)));
  assert.deepEqual(await api.askStream({ question: '问题' }, { onEvent: e => types.push(e.type) }), rejected);
  assert.deepEqual(types, ['delta', 'reset', 'delta', 'result']);
});

test('mid-stream failure retains error code and cannot resolve a partial answer', async t => {
  t.mock.method(globalThis, 'fetch', async () => response(event('delta', { text: '草稿' })
    + 'event: error\ndata: ' + JSON.stringify({ success: false, data: null,
      error: { code: 'generation_unavailable', message: '模型连接中断' } }) + '\n\n'));
  await assert.rejects(api.askStream({ question: '问题' }), e => e instanceof ApiError && e.code === 'generation_unavailable');
});

test('premature EOF is distinguished from a completed response', async t => {
  t.mock.method(globalThis, 'fetch', async () => response(event('delta', { text: '半截回答' })));
  await assert.rejects(api.askStream({ question: '问题' }), e => e.code === 'stream_incomplete');
});

test('busy HTTP error retains unified API failure', async t => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ success: false, data: null,
    error: { code: 'service_busy', message: '稍后重试' } }, { status: 503 }));
  await assert.rejects(api.askStream({ question: '问题' }), e => e.code === 'service_busy');
});

test('JSON endpoint, invalid event and invalid terminal payload fail explicitly', async t => {
  for (const value of [
    Response.json(envelope(final)),
    response('event: delta\ndata: {bad}\n\n'),
    response(event('delta', { text: 3 })),
    response(event('result', { answer: 'x' })),
    response(event('stage', { stage: 'unknown' })),
  ]) {
    const mock = t.mock.method(globalThis, 'fetch', async () => value);
    await assert.rejects(api.askStream({ question: '问题' }), e => e.code === 'invalid_response');
    mock.mock.restore();
  }
});

test('explicit abort clears wait and differs from timeout', async t => {
  const controller = new AbortController();
  let started;
  const ready = new Promise(resolve => { started = resolve; });
  t.mock.method(globalThis, 'fetch', async (_url, options) => new Promise((_resolve, reject) => {
    options.signal.addEventListener('abort', () => reject(new DOMException('stop', 'AbortError')), { once: true });
    started();
  }));
  const pending = api.askStream({ question: '问题' }, { signal: controller.signal });
  await ready;
  controller.abort();
  await assert.rejects(pending, e => e.code === 'request_cancelled');
  await assert.rejects(api.askStream({ question: '问题' }, { timeout: 1 }), e => e.code === 'request_timeout');
});
