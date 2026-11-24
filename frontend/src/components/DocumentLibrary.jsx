import { useRef, useState } from 'react';
import { FILE_ACCEPT } from '../file-formats';
import { dateTime, fileType, number, shortId, states } from '../utils';
import UploadQueue from './UploadQueue';
import { EmptyState, Icon, StatusBadge } from './ui';

export default function DocumentLibrary({ documents, health, loading, busy, onUpload, onReindex, onDelete, onDetail, onRefresh, onSearch, uploadQueue = [], batchActive = false, stoppingUploads, onStopUploads, onRetryUploads, onResumeUploads, onClearUploads }) {
  const fileInput = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [filter, setFilter] = useState('');
  const [status, setStatus] = useState('all');
  const shown = documents.filter(doc => `${doc.filename} ${doc.document_id}`.toLowerCase().includes(filter.toLowerCase()) && (status === 'all' || doc.status === status));
  const maxSize = health?.max_file_bytes ? Math.round(health.max_file_bytes / 1024 / 1024) : 20;
  function choose(files) { if (files.length) onUpload([...files]); }
  return <>
    <div className="page-heading"><div><p className="eyebrow">YOUR LOCAL KNOWLEDGE</p><h1>文档库<span className="heading-dot">.</span></h1><p className="page-description">让文档成为可检索的知识，从上传第一份文件开始。</p></div>
      <button className="button primary" disabled={busy} onClick={() => fileInput.current?.click()}><Icon name="upload" size={17}/>批量导入</button></div>
    <div className="stats-strip"><div><span className="stat-label">全部文档</span><strong>{number(documents.length)}<small>份</small></strong></div>
      <div><span className="stat-label">已就绪，可检索</span><strong>{number(documents.filter(doc => doc.status === 'indexed').length)}<small>份</small></strong></div>
      <div><span className="stat-label">已索引文本块</span><strong>{number(documents.filter(doc => doc.status === 'indexed').reduce((sum, doc) => sum + doc.chunk_count, 0))}<small>块</small></strong></div>
      <div className="stats-model"><span className="model-symbol"><Icon name="layers" size={23}/></span><div><b>{health?.embedding_model || 'BGE-M3'}</b><span>本地向量模型配置</span></div></div></div>
    <input ref={fileInput} className="visually-hidden" type="file" multiple accept={health?.supported_types?.map(type => `.${type}`).join(',') || FILE_ACCEPT} aria-label="选择上传文档（支持多选）" tabIndex={-1} disabled={busy}
      onChange={event => { choose(event.target.files); event.target.value = ''; }}/>
    <div className={`dropzone ${dragging ? 'dragging' : ''} ${busy ? 'disabled' : ''}`}
      onDragOver={event => { event.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); }}
      onDrop={event => { event.preventDefault(); setDragging(false); if (!busy) choose(event.dataTransfer.files); }}>
      <span className="upload-art"><Icon name="upload" size={27}/></span>
      <div className="dropzone-copy"><h2>将多个文档拖到这里</h2><p>支持 PDF、Word、Excel、CSV/TSV、文本和图片 OCR · 单文件最大 {maxSize} MB</p></div>
      <button className="button secondary" disabled={busy} onClick={() => fileInput.current?.click()}>选择文件（可多选）<Icon name="arrow" size={16}/></button>
    </div>
    <UploadQueue items={uploadQueue} active={batchActive} stopping={stoppingUploads} onStop={onStopUploads}
      onRetry={onRetryUploads} onResume={onResumeUploads} onClear={onClearUploads} onDetail={onDetail}/>
    <section className="documents-panel" aria-label="文档列表">
      <div className="section-heading"><h2>所有文档 <span>{documents.length}</span></h2><button className="icon-button" onClick={onRefresh} disabled={loading || busy} aria-label="刷新文档列表" title="刷新列表"><Icon name="refresh" size={18} className={loading ? 'rotate' : ''}/></button></div>
      <div className="table-toolbar"><label className="filter-input"><Icon name="search" size={17}/><input placeholder="搜索文件名或文档 ID…" aria-label="搜索文件名或文档 ID" value={filter} onChange={event => setFilter(event.target.value)}/></label>
        <select aria-label="筛选处理状态" value={status} onChange={event => setStatus(event.target.value)}><option value="all">全部状态</option>{Object.entries(states).filter(([key]) => key !== 'deleted').map(([key, [label]]) => <option key={key} value={key}>{label}</option>)}</select></div>
      <div className="table-scroll"><table><thead><tr><th>文件名称</th><th>状态</th><th>文本块</th><th>上传时间</th><th className="actions-heading">操作</th></tr></thead>
        <tbody>{shown.map(doc => <tr key={doc.document_id}>
          <td><button className="file-cell" onClick={() => onDetail(doc.document_id)}><span className={`file-type type-${fileType(doc.filename).toLowerCase()}`}><Icon name="file" size={21}/></span><span><b>{doc.filename}</b><small>{fileType(doc.filename)}<span>·</span><span className="mono">{shortId(doc.document_id)}</span></small></span></button></td>
          <td><StatusBadge status={doc.status}/></td><td className="numeric">{number(doc.chunk_count)}</td><td className="date-cell">{dateTime(doc.uploaded_at)}</td>
          <td><div className="row-actions"><button className="icon-button" aria-label={`查看 ${doc.filename} ${shortId(doc.document_id)}`} title="查看文本块" onClick={() => onDetail(doc.document_id)}><Icon name="file" size={17}/></button>
            <button className="icon-button" aria-label={`重新索引 ${doc.filename} ${shortId(doc.document_id)}`} title="重新索引" disabled={busy || ['deleting','delete_failed','indexing'].includes(doc.status)} onClick={() => onReindex(doc)}><Icon name="refresh" size={17}/></button>
            <button className="icon-button danger-hover" aria-label={`删除 ${doc.filename} ${shortId(doc.document_id)}`} title="删除文档" disabled={busy} onClick={() => onDelete(doc)}><Icon name="trash" size={17}/></button></div></td>
        </tr>)}</tbody></table></div>
      {loading && !documents.length ? <div className="loading-state"><span className="spinner"/>正在读取文档库…</div> : !shown.length && <EmptyState title={documents.length ? '没有匹配的文档' : '知识库从这里开始'}>{documents.length ? '换一个关键词或处理状态试试。' : '上传一份文档，系统会自动提取文字并建立索引。'}</EmptyState>}
      <div className="table-footer"><span>共 {shown.length} 份文档 · 保存在本机</span><button className="text-link" onClick={onSearch}>去检索知识<Icon name="arrow" size={15}/></button></div>
    </section>
    <div className="local-note"><Icon name="info" size={16}/><p>文档在本机处理，索引完成后即可检索。点击文件名，可以查看原文分块和来源信息。</p></div>
  </>;
}
