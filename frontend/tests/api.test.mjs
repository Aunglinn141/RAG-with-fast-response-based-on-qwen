import assert from 'node:assert/strict';
import test from 'node:test';
import { api, ApiError } from '../src/api.js';

function successResponse(data, status = 200) {
  return Response.json({ success: true, data, error: null }, { status });
}

function failureResponse(error, status = 503) {
  return Response.json({ success: false, data: null, error }, { status });
}

test('upload sends a single multipart file without overriding its boundary', async t => {
  let request;
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    request = { url, options };
    return successResponse({ document_id: 'new-document', status: 'indexed' }, 201);
  });
  const file = new File(['sample text'], 'sample.txt', { type: 'text/plain' });
  const result = await api.upload(file);
  assert.equal(request.url, '/api/documents');
  assert.equal(request.options.method, 'POST');
  assert.equal(new Headers(request.options.headers).has('content-type'), false);
  assert.deepEqual([...request.options.body.keys()], ['file']);
  assert.equal(request.options.body.get('file').name, 'sample.txt');
  assert.equal(await request.options.body.get('file').text(), 'sample text');
  assert.equal(result.status, 'indexed');
});

test('index failure preserves the saved document ID for recovery', async t => {
  t.mock.method(globalThis, 'fetch', async () => failureResponse({
    code: 'index_failed', message: 'Saved, but indexing failed.', document_id: 'recover-this-id',
  }));
  await assert.rejects(api.upload(new File(['text'], 'retry.txt')), error => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.code, 'index_failed');
    assert.equal(error.documentId, 'recover-this-id');
    assert.equal(error.message, 'Saved, but indexing failed.');
    return true;
  });
});

test('search sends UTF-8 JSON with the selected document and top_k', async t => {
  const query = { query: '\u6843\u56ed\u7ed3\u4e49', top_k: 3, document_id: 'chosen-document' };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, '/api/search');
    assert.equal(options.method, 'POST');
    assert.equal(new Headers(options.headers).get('content-type'), 'application/json');
    assert.deepEqual(JSON.parse(options.body), query);
    return successResponse({ query: query.query, matches: [] });
  });
  assert.deepEqual((await api.search(query)).matches, []);
});

test('queued search can be stopped while waiting for response headers', async t => {
  const controller = new AbortController();
  t.mock.method(globalThis, 'fetch', (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  }));
  const pending = api.search({ query: '问题' }, { signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, error => error.code === 'request_cancelled');
});

test('backend validation details remain actionable', async t => {
  t.mock.method(globalThis, 'fetch', async () => failureResponse({
    code: 'validation_error', message: '请求参数无效', details: [
      { loc: ['body', 'query'], msg: 'Query cannot be empty', type: 'value_error' },
      { loc: ['body', 'top_k'], msg: 'top_k must be between 1 and 50', type: 'greater_than_equal' },
    ],
  }, 422));
  await assert.rejects(api.search({ query: '', top_k: 0 }), error => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.code, 'validation_error');
    assert.equal(error.details.length, 2);
    assert.deepEqual(error.details[0].loc, ['body', 'query']);
    assert.match(error.message, /Query cannot be empty/);
    assert.match(error.message, /top_k must be between 1 and 50/);
    return true;
  });
});

test('JSON chat preserves explicitly supplied conversation history', async t => {
  const payload = { question: '它在哪里？', history: [{ question: '介绍测试壶', answer: '这是一件青铜器。' }] };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, '/api/chat');
    assert.deepEqual(JSON.parse(options.body), payload);
    return successResponse({ answer: '测试博物馆。', status: 'answered' });
  });
  await api.ask(payload);
});

test('knowledge question discards document scope and retrieval tuning before sending', async t => {
  const question = { question: '刘备、关羽和张飞是如何结义的？', top_k: 4, document_id: 'chosen-document' };
  const response = {
    question: question.question, answer: '三人在桃园结义。', status: 'answered',
    sources: [{ source_id: 'S1', chunk_id: 'chosen-chunk', text: '三人于桃园结义。',
      metadata: { document_id: 'chosen-document', filename: '三国演义.txt', chunk_index: 4 }, cited: true }],
    evidence: [{ source_id: 'S1', quote: '三人于桃园结义。' }], warnings: [],
  };
  let timeout;
  t.mock.method(globalThis, 'setTimeout', (callback, duration) => { timeout = duration; return 123; });
  t.mock.method(globalThis, 'clearTimeout', handle => assert.equal(handle, 123));
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(timeout, undefined, 'queued chat waits for response headers before starting its body timeout');
    assert.equal(url, '/api/chat');
    assert.equal(options.method, 'POST');
    assert.equal(new Headers(options.headers).get('content-type'), 'application/json');
    assert.deepEqual(JSON.parse(options.body), { question: question.question,
      ...(question.model === undefined ? {} : { model: question.model }) });
    return successResponse(response);
  });
  assert.deepEqual(await api.ask(question), response);
  assert.equal(timeout, 6 * 60 * 1000);
});

test('insufficient evidence is an answer result and does not invent a source', async t => {
  const response = { question: '资料中没有记录的事情？', answer: '根据现有资料无法确定。',
    status: 'insufficient_evidence', sources: [], evidence: [], warnings: [] };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(Object.hasOwn(JSON.parse(options.body), 'document_id'), false);
    return successResponse(response);
  });
  assert.deepEqual(await api.ask({ question: response.question, top_k: 4 }), response);
});

test('local answer generation failure preserves the backend error for the user', async t => {
  t.mock.method(globalThis, 'fetch', async () => failureResponse({
    code: 'generation_unavailable', message: '本地问答模型未就绪，请检查 Qwen 模型。',
  }));
  await assert.rejects(api.ask({ question: '问题', top_k: 4 }), error => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.code, 'generation_unavailable');
    assert.equal(error.message, '本地问答模型未就绪，请检查 Qwen 模型。');
    return true;
  });
});

test('connection failure has its own error code', async t => {
  t.mock.method(globalThis, 'fetch', async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(api.health(), error => error instanceof ApiError && error.code === 'connection_error');
});

test('request timeout aborts fetch and releases its timer', async t => {
  let cleared = false;
  let aborted = false;
  t.mock.method(globalThis, 'setTimeout', callback => { queueMicrotask(callback); return 42; });
  t.mock.method(globalThis, 'clearTimeout', handle => { assert.equal(handle, 42); cleared = true; });
  t.mock.method(globalThis, 'fetch', (url, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => {
      aborted = true;
      reject(new DOMException('Aborted', 'AbortError'));
    }, { once: true });
  }));
  await assert.rejects(api.health(), error => error instanceof ApiError && error.code === 'request_timeout');
  assert.equal(aborted, true);
  assert.equal(cleared, true);
});

test('timeout while reading response body is not mislabeled as invalid data', async t => {
  t.mock.method(globalThis, 'fetch', async () => ({
    ok: true, status: 200,
    json: async () => { throw new DOMException('Body aborted', 'AbortError'); },
  }));
  await assert.rejects(api.document('sample'), error => error instanceof ApiError && error.code === 'request_timeout');
});

test('malformed successful responses are reported as invalid data', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('<html>not the API</html>', { status: 200 }));
  await assert.rejects(api.documents(), error => error instanceof ApiError && error.code === 'invalid_response');
});

test('delete targets exactly the selected ID and does not use a request body', async t => {
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, '/api/documents/chosen%2Fdocument');
    assert.equal(options.method, 'DELETE');
    assert.equal(options.body, undefined);
    return successResponse({ document_id: 'chosen/document', status: 'deleted', remaining_vectors: 0 });
  });
  assert.equal((await api.remove('chosen/document')).remaining_vectors, 0);
});

test('health unwraps only the data payload for existing GUI components', async t => {
  const health = { status: 'ok', embedding_model: 'bge-m3', chat_model: 'qwen3:1.7b' };
  t.mock.method(globalThis, 'fetch', async () => successResponse(health));
  assert.deepEqual(await api.health(), health);
});

test('a failure envelope is not mistaken for success even with HTTP 200', async t => {
  t.mock.method(globalThis, 'fetch', async () => failureResponse({
    code: 'backend_unavailable', message: 'API 暂时无法连接。',
  }, 200));
  await assert.rejects(api.health(), error => error instanceof ApiError
    && error.code === 'backend_unavailable' && error.message === 'API 暂时无法连接。');
});

test('old or inconsistent envelopes fail explicitly instead of leaking into the GUI', async t => {
  for (const [body, status] of [
    [{ status: 'ok' }, 200],
    [{ success: true, data: {} }, 200],
    [{ success: 'true', data: {}, error: null }, 200],
    [{ success: true, data: null, error: null }, 200],
    [{ success: true, data: {}, error: { code: 'conflict', message: 'Error' } }, 200],
    [{ success: true, data: {}, error: null }, 503],
    [{ success: false, data: {}, error: { code: 'error', message: 'Error' } }, 503],
    [{ success: false, data: null, error: null }, 503],
    [{ success: false, data: null, error: { code: 'error' } }, 503],
  ]) {
    const mock = t.mock.method(globalThis, 'fetch', async () => Response.json(body, { status }));
    await assert.rejects(api.health(), error => error instanceof ApiError && error.code === 'invalid_response');
    mock.mock.restore();
  }
});

test('model catalog is read from the project API without loading a model', async t => {
  const catalog = { default_model: 'qwen3:1.7b', models: [{ name: 'gemma3:1b' }], unavailable_models: [] };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, '/api/models');
    assert.equal(options.method, undefined);
    return successResponse(catalog);
  });
  assert.deepEqual(await api.models(), catalog);
});

test('question sends the explicitly selected local model', async t => {
  const question = { question: '问题', model: 'gemma3:1b', top_k: 4 };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, '/api/chat');
    assert.deepEqual(JSON.parse(options.body), { question: question.question,
      ...(question.model === undefined ? {} : { model: question.model }) });
    return successResponse({ answer: '回答', model: { name: question.model } });
  });
  assert.equal((await api.ask(question)).model.name, 'gemma3:1b');
});

test('removed model is reported without silently selecting another model', async t => {
  t.mock.method(globalThis, 'fetch', async () => failureResponse({
    code: 'invalid_chat_model', message: '项目中没有所选模型。',
  }, 422));
  await assert.rejects(api.ask({ question: '问题', model: 'missing' }),
    error => error.code === 'invalid_chat_model');
});
