import assert from 'node:assert/strict';
import test from 'node:test';
import { createUploadQueue, runUploadQueue, prepareUploadRetry, summarizeUploads } from '../src/upload-queue.js';

const file = (name, text = 'content') => new File([text], name);
const success = id => ({ document_id: id, status: 'indexed', chunk_count: 3 });

test('mixed batch validates each file and imports the remaining valid files', async () => {
  const rows = createUploadQueue([file('a.txt'), file('bad.exe'), file('empty.md', ''), file('large.pdf', 'x'.repeat(21)), file('b.DOCX')], 20);
  const called = [];
  const final = await runUploadQueue(rows, { upload: async f => { called.push(f.name); return success(f.name); } });
  assert.deepEqual(called, ['a.txt', 'b.DOCX']);
  assert.deepEqual(final.map(r => r.status), ['indexed', 'invalid', 'invalid', 'invalid', 'indexed']);
  assert.deepEqual(summarizeUploads(final), { total: 5, indexed: 2, saved: 0, failed: 3, uncertain: 0, remaining: 0, handled: 5 });
});

test('upload requests are serial and parsing failure does not abort following files', async () => {
  let active = 0, maximum = 0;
  const calls = [];
  const final = await runUploadQueue(createUploadQueue([file('a.txt'), file('broken.pdf'), file('b.md')]), {
    upload: async f => {
      active++; maximum = Math.max(maximum, active); calls.push(f.name);
      await new Promise(resolve => setTimeout(resolve, 4)); active--;
      if (f.name === 'broken.pdf') throw { code: 'invalid_document', message: 'PDF cannot be parsed' };
      return success(f.name);
    },
  });
  assert.equal(maximum, 1);
  assert.deepEqual(calls, ['a.txt', 'broken.pdf', 'b.md']);
  assert.deepEqual(final.map(r => r.status), ['indexed', 'failed', 'indexed']);
});

test('stop finishes the active file; resume never uploads completed files again', async () => {
  let stop = false;
  const calls = [];
  const upload = async f => { calls.push(f.name); stop = true; return success(f.name); };
  const paused = await runUploadQueue(createUploadQueue([file('a.txt'), file('b.txt'), file('c.txt')]), { upload, shouldStop: () => stop });
  assert.deepEqual(paused.map(r => r.status), ['indexed', 'paused', 'paused']);
  assert.deepEqual(calls, ['a.txt']);
  const resumed = await runUploadQueue(prepareUploadRetry(paused, 'resume'), { upload });
  assert.deepEqual(calls, ['a.txt', 'b.txt', 'c.txt']);
  assert.equal(summarizeUploads(resumed).indexed, 3);
});

test('index failure retry reuses the saved document ID rather than uploading again', async () => {
  let uploads = 0;
  const initial = await runUploadQueue(createUploadQueue([file('book.txt')]), {
    upload: async () => { uploads++; throw { code: 'index_failed', message: 'Model unavailable', documentId: 'saved-id' }; },
  });
  let reindexed;
  const final = await runUploadQueue(prepareUploadRetry(initial, 'retry'), {
    upload: async () => { uploads++; throw new Error('must not upload again'); },
    reindex: async id => { reindexed = id; return success(id); },
  });
  assert.equal(uploads, 1); assert.equal(reindexed, 'saved-id'); assert.equal(final[0].status, 'indexed');
});

for (const code of ['request_timeout', 'connection_error', 'invalid_response', 'backend_unavailable']) {
  test(code + ' marks the outcome uncertain, pauses pending files and is not blindly retried', async () => {
    let calls = 0;
    const final = await runUploadQueue(createUploadQueue([file('a.txt'), file('b.txt')]), {
      upload: async () => { calls++; throw { code, message: 'Request failed' }; },
    });
    assert.equal(calls, 1);
    assert.deepEqual(final.map(r => r.status), ['uncertain', 'paused']);
    assert.deepEqual(prepareUploadRetry(final, 'retry').map(r => r.status), ['uncertain', 'paused']);
  });
}

test('server busy pauses the queue, while a later retry can safely retry the rejected file', async () => {
  const final = await runUploadQueue(createUploadQueue([file('a.txt'), file('b.txt')]), {
    upload: async () => { throw { code: 'service_busy', message: 'Busy' }; },
  });
  assert.deepEqual(final.map(r => r.status), ['failed', 'paused']);
  assert.equal(prepareUploadRetry(final, 'retry')[0].status, 'queued');
});

test('auto indexing disabled is shown as saved, not searchable success', async () => {
  const final = await runUploadQueue(createUploadQueue([file('a.txt')]), {
    upload: async () => ({ document_id: 'ready-id', status: 'ready', chunk_count: 2 }),
  });
  assert.equal(final[0].status, 'saved');
  assert.equal(summarizeUploads(final).indexed, 0);
  assert.equal(summarizeUploads(final).saved, 1);
});

test('same names remain distinct entries and receive their own document IDs', async () => {
  let count = 0;
  const final = await runUploadQueue(createUploadQueue([file('same.txt', 'first'), file('same.txt', 'second')]), {
    upload: async () => success(String(++count)),
  });
  assert.equal(count, 2);
  assert.notEqual(final[0].id, final[1].id);
  assert.notEqual(final[0].documentId, final[1].documentId);
});

test('OCR and spreadsheet uploads share the serial queue while unsupported formats remain rejected', () => {
  const valid = ['scan.PDF', 'data.XLSX', 'data.xls', 'data.xlsm', 'data.csv', 'data.tsv', 'scan.png', 'scan.JPG', 'scan.tiff', 'page.html', 'info.json'];
  assert.ok(createUploadQueue(valid.map(name => file(name))).every(row => row.status === 'queued'));
  assert.ok(createUploadQueue(['old.doc', 'book.xlsb', 'archive.zip'].map(name => file(name))).every(row => row.status === 'invalid'));
});
