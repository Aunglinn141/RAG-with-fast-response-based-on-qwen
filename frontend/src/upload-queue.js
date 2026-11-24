// One request at a time: preserve the API's indexing and memory limits.
import { supportedFilename } from "./file-formats.js";
export const DEFAULT_FILE_LIMIT = 20 * 1024 * 1024;
const uncertainCodes = new Set(['request_timeout', 'connection_error', 'invalid_response', 'backend_unavailable', 'internal_error']);
const pauseCodes = new Set([...uncertainCodes, 'service_busy', 'index_busy']);

export function createUploadQueue(files, maxBytes = DEFAULT_FILE_LIMIT) {
  return Array.from(files, (file, index) => {
    const message = !supportedFilename(file.name) ? '不支持此格式，请选择 PDF、Word、Excel、文本或 OCR 图片。'
      : !file.size ? '文件为空，请选择包含文字的文件。'
      : file.size > maxBytes ? `超过单文件 ${(maxBytes / 1024 / 1024).toFixed(0)} MiB 上限。` : '';
    return { id: String(index), file, filename: file.name, size: file.size,
      status: message ? 'invalid' : 'queued', message, documentId: null };
  });
}

export function summarizeUploads(items) {
  const counts = { total: items.length, indexed: 0, saved: 0, failed: 0, uncertain: 0, remaining: 0, handled: 0 };
  for (const item of items) {
    if (item.status === 'indexed' || item.status === 'saved') counts[item.status]++;
    else if (item.status === 'failed' || item.status === 'invalid') counts.failed++;
    else if (item.status === 'uncertain') counts.uncertain++;
    else counts.remaining++;
  }
  counts.handled = counts.total - counts.remaining;
  return counts;
}

export function prepareUploadRetry(items, mode) {
  const target = mode === 'resume' ? 'paused' : 'failed';
  return items.map(item => item.status === target ? { ...item, status: 'queued', message: '' } : item);
}

export async function runUploadQueue(initial, { upload, reindex, onUpdate = () => {}, shouldStop = () => false }) {
  let items = initial.map(item => ({ ...item }));
  const emit = () => onUpdate(items.map(item => ({ ...item })));
  let pauseMessage = '';
  emit();
  for (let index = 0; index < items.length; index++) {
    if (items[index].status !== 'queued') continue;
    if (shouldStop() || pauseMessage) break;
    const item = items[index];
    items[index] = { ...item, status: 'processing', message: item.documentId ? '正在重新索引…' : '正在上传并处理…' };
    emit();
    try {
      // Index failures already have a document: retry its index, never re-upload it.
      const result = item.documentId ? await reindex(item.documentId) : await upload(item.file);
      items[index] = { ...item, file: null, documentId: result.document_id,
        status: result.status === 'indexed' ? 'indexed' : 'saved',
        message: result.status === 'indexed' ? `${result.chunk_count} 个文本块可检索` : '文件已保存，等待建立索引。' };
    } catch (error) {
      const uncertain = !error?.code || uncertainCodes.has(error.code);
      items[index] = { ...item, documentId: error?.documentId || item.documentId,
        status: uncertain ? 'uncertain' : 'failed',
        message: (error?.message || '导入请求未完成。') + (uncertain ? ' 请先核对文档列表，避免重复上传。' : ''),
        code: error?.code };
      if (uncertain || pauseCodes.has(error.code)) pauseMessage = '服务状态需要确认，尚未开始导入。';
    }
    emit();
  }
  items = items.map(item => item.status === 'queued' ? { ...item, status: 'paused',
    message: pauseMessage || '已停止，尚未开始导入。' } : item);
  emit();
  return items;
}
