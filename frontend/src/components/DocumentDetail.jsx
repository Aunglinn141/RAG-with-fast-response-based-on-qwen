import { useEffect, useState } from 'react';
import { api, downloadUrl } from '../api';
import { dateTime, number } from '../utils';
import { EmptyState, Icon, Modal, StatusBadge } from './ui';

export default function DocumentDetail({ id, initialChunk, revision, onClose }) {
  const [record, setRecord] = useState(null);
  const [error, setError] = useState('');
  const [page, setPage] = useState(initialChunk ? Math.floor((initialChunk - 1) / 5) : 0);
  useEffect(() => {
    let active = true;
    setError('');
    setRecord(null);
    api.document(id).then(data => { if (active) setRecord(data); }).catch(error => { if (active) setError(error.message); });
    return () => { active = false; };
  }, [id, revision]);
  return <Modal title="文档详情" drawer onClose={onClose}>
    {error ? <div role="alert" className="inline-error">{error}</div> : !record ? <div className="loading-state"><span className="spinner"/>正在读取文档…</div> : <>
      <div className="detail-title"><span className="file-icon"><Icon name="file" size={27}/></span>
        <h3>{record.filename}</h3><StatusBadge status={record.status}/></div>
      <div className="detail-downloads"><a className="button secondary" href={downloadUrl(id)}><Icon name="download" size={16}/>下载原文件</a>
        <a className="text-link" href={downloadUrl(id, 'text')}>下载纯文本<Icon name="download" size={14}/></a></div>
      <dl className="detail-facts"><div><dt>上传时间</dt><dd>{dateTime(record.created_at)}</dd></div>
        <div><dt>文本块</dt><dd>{number(record.chunk_count)} 块 · {number(record.character_count)} 字符</dd></div>
        <div><dt>文档 ID</dt><dd className="mono break-all">{id}</dd></div>
        <div><dt>向量模型</dt><dd>{record.index?.model_name || '尚未索引'}{record.index?.embedding_dimension ? ` · ${record.index.embedding_dimension} 维` : ''}</dd></div>
        {record.index?.model_version && <div><dt>模型版本</dt><dd className="mono break-all">{record.index.model_version}</dd></div>}
      </dl>
      {record.index?.last_error && <div className="inline-error" role="alert">最近处理信息：{record.index.last_error}</div>}
      {record.warnings?.map((warning, index) => <p className="inline-warning" key={index}>{warning}</p>)}
      <div className="chunk-header"><h3>文本分块</h3><span>{page * 5 + 1}–{Math.min((page + 1) * 5, record.chunk_count)} / {number(record.chunk_count)}</span></div>
      <div className="chunk-list">{record.chunks.slice(page * 5, page * 5 + 5).map(chunk => <article key={chunk.chunk_id} className={`chunk-card ${chunk.metadata.chunk_index === initialChunk ? 'highlighted' : ''}`}>
        <div className="chunk-meta"><span>分块 {chunk.metadata.chunk_index}</span>
          {chunk.metadata.page_number && <span>第 {chunk.metadata.page_number} 页</span>}
          {chunk.metadata.row_start && <span>第 {chunk.metadata.row_start}–{chunk.metadata.row_end} 行</span>}
          {chunk.metadata.extraction_method === 'ocr' && <span>OCR 识别</span>}
          {chunk.metadata.section_title && <span>{chunk.metadata.section_title}</span>}</div>
        <p>{chunk.text}</p>
        <div className="chunk-offset">字符 {chunk.metadata.start_index}–{chunk.metadata.end_index}
          {chunk.metadata.cell_ranges?.length ? ` · 单元格 ${chunk.metadata.cell_ranges.join('、')}` : chunk.metadata.paragraph_start && ` · 段落 ${chunk.metadata.paragraph_start}–${chunk.metadata.paragraph_end}`}</div>
      </article>)}</div>
      {!record.chunks.length && <EmptyState title="暂无文本块">此文档尚未生成可查看的文本块。</EmptyState>}
      <div className="pagination"><button className="button secondary" disabled={page === 0} onClick={() => setPage(p => p - 1)}>上一页</button>
        <span>{page + 1} / {Math.max(1, Math.ceil(record.chunk_count / 5))}</span>
        <button className="button secondary" disabled={(page + 1) * 5 >= record.chunk_count} onClick={() => setPage(p => p + 1)}>下一页</button></div>
    </>}
  </Modal>;
}
