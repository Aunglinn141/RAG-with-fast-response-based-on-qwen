import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api';
import { duration } from './utils';
import { Icon, Modal, Notice } from './components/ui';
import DocumentLibrary from './components/DocumentLibrary';
import DocumentDetail from './components/DocumentDetail';
import SearchWorkspace from './components/SearchWorkspace';
import { createUploadQueue, runUploadQueue, prepareUploadRetry, summarizeUploads } from './upload-queue';
import { newConversation, loadConversations, saveConversations } from './conversations';

export default function App() {
  const [savedChats] = useState(() => loadConversations());
  const [view, setView] = useState('chat');
  const [documents, setDocuments] = useState([]);
  const [health, setHealth] = useState(null);
  const [connection, setConnection] = useState('checking');
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState(() => savedChats.error ? { type: 'error', message: savedChats.error } : null);
  const [chatStorageError, setChatStorageError] = useState('');
  const [operation, setOperation] = useState(null);
  const [seconds, setSeconds] = useState(0);
  const [searchBusy, setSearchBusy] = useState(false);
  const [revision, setRevision] = useState(0);
  const [detail, setDetail] = useState(null);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [uploadQueue, setUploadQueue] = useState([]);
  const [stoppingUploads, setStoppingUploads] = useState(false);
  const uploadStop = useRef(false), operationLock = useRef(false);
  const [sidebarOpen, setSidebarOpen] = useState(() => typeof window === 'undefined' || window.innerWidth > 760);
  const [conversations, setConversations] = useState(() => savedChats.conversations.length ? savedChats.conversations : [newConversation()]);
  const [activeConversation, setActiveConversation] = useState(savedChats.activeId);
  const conversation = conversations.find(item => item.id === activeConversation) || conversations[0];
  const refreshSequence = useRef(0);
  const busy = !!operation || searchBusy;

  useEffect(() => {
    setChatStorageError(saveConversations(conversations, conversation.id));
  }, [conversations, conversation.id]);

  const checkHealth = useCallback(async () => {
    try { const data = await api.health(); setHealth(data); setConnection('online'); }
    catch { setConnection('offline'); }
  }, []);

  const refresh = useCallback(async (silent = false) => {
    const sequence = ++refreshSequence.current;
    if (!silent) setLoading(true);
    try {
      const data = await api.documents();
      if (sequence === refreshSequence.current) setDocuments([...data.documents].sort((a, b) => b.uploaded_at.localeCompare(a.uploaded_at)));
    } catch (error) {
      if (sequence === refreshSequence.current && !silent) setNotice({ type: 'error', message: error.message });
    } finally { if (sequence === refreshSequence.current) setLoading(false); }
  }, []);

  useEffect(() => { checkHealth(); if (view !== 'chat') refresh(); }, [checkHealth, refresh, view]);
  useEffect(() => {
    const timer = setInterval(() => { checkHealth(); if (!busy && view !== 'chat') refresh(true); }, 15000);
    return () => clearInterval(timer);
  }, [busy, checkHealth, refresh, view]);
  useEffect(() => {
    if (!operation || operation.kind === 'batch') return;
    setSeconds(0);
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - operation.started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [operation]);

  useEffect(() => {
    if (operation?.kind !== 'batch') return;
    const protect = event => { event.preventDefault(); event.returnValue = ''; };
    window.addEventListener('beforeunload', protect);
    return () => window.removeEventListener('beforeunload', protect);
  }, [operation]);
  useEffect(() => {
    const escape = event => { if (event.key === 'Escape' && window.innerWidth <= 760) setSidebarOpen(false); };
    window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, []);
  const showError = useCallback(error => setNotice({ type: 'error', message: error.message }), []);
  const closeDetail = useCallback(() => setDetail(null), []);
  const closeDelete = useCallback(() => setDeleteTarget(null), []);
  function openDetail(id, initialChunk) { setDetail({ id, initialChunk }); }
  function changeView(value) {
    if (busy || operationLock.current) return;
    setNotice(null); setDetail(null); setDeleteTarget(null); setView(value);
    if (window.innerWidth <= 760) setSidebarOpen(false);
  }

  async function mutate(kind, filename, action, id) {
    if (busy || operationLock.current) return;
    operationLock.current = true;
    setNotice(null); setRevision(value => value + 1);
    setOperation({ kind, filename, id, started: Date.now() });
    try {
      const result = await action();
      const message = kind === 'delete' ? `已删除「${filename}」，对应向量已清理。`
        : result.status === 'indexed' ? `「${filename}」已完成索引，${result.chunk_count} 个文本块可以检索。`
        : `「${filename}」已保存并生成文本块，点击重新索引后即可检索。`;
      if (kind === 'delete') {
        setDocuments(items => items.filter(doc => doc.document_id !== id));
        if (detail?.id === id) setDetail(null);
      }
      setNotice({ type: 'success', message });
    } catch (error) {
      setNotice({ type: 'error', message: error.message + (error.documentId ? ` 文档 ID：${error.documentId}` : '') });
    } finally {
      setRevision(value => value + 1);
      await refresh(true);
      operationLock.current = false; setOperation(null);
      checkHealth();
    }
  }

  async function startUploads(items) {
    if (busy || operationLock.current) return;
    operationLock.current = true; uploadStop.current = false; setStoppingUploads(false);
    setNotice(null); setUploadQueue(items); setOperation({ kind: 'batch', started: Date.now() });
    setRevision(value => value + 1);
    try {
      const final = await runUploadQueue(items, { upload: api.upload, reindex: api.reindex,
        onUpdate: setUploadQueue, shouldStop: () => uploadStop.current });
      const counts = summarizeUploads(final);
      setNotice({ type: counts.failed || counts.uncertain ? 'error' : counts.remaining ? 'info' : 'success',
        message: `本批导入：已索引 ${counts.indexed} 份，待索引 ${counts.saved} 份，失败 ${counts.failed} 份，待核对 ${counts.uncertain} 份，未导入 ${counts.remaining} 份。` });
    } finally {
      await refresh(true);
      operationLock.current = false; setOperation(null); setStoppingUploads(false);
      setRevision(value => value + 1); checkHealth();
    }
  }
  function upload(files) {
    if (busy || operationLock.current || !files.length) return;
    startUploads(createUploadQueue(files, health?.max_file_bytes));
  }
  function startConversation() {
    if (busy || operationLock.current) return;
    const item = newConversation();
    setConversations(items => [item, ...items.filter(chat => chat.messages.length)]);
    setActiveConversation(item.id); changeView('chat');
  }
  function chooseConversation(id) {
    if (busy) return;
    setActiveConversation(id); changeView('chat');
  }
  function removeConversation(id) {
    if (busy) return;
    const rest = conversations.filter(item => item.id !== id);
    if (!rest.length) rest.push(newConversation());
    setConversations(rest);
    if (conversation.id === id) setActiveConversation(rest[0].id);
  }
  function updateMessages(update) {
    const id = conversation.id;
    setConversations(items => items.map(item => item.id === id ? { ...item,
      messages: typeof update === 'function' ? update(item.messages) : update } : item));
  }

  const operationNames = { upload: '上传并处理', reindex: '重新索引', delete: '删除' };
  return <div className={`app-shell chat-shell ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
    {sidebarOpen && <button className="sidebar-scrim" aria-label="关闭导航" onClick={() => setSidebarOpen(false)}/>}
    <aside className="sidebar" id="workspace-sidebar" aria-label="工作空间导航">
      <div className="sidebar-brand"><a className="brand" href="#" onClick={event => { event.preventDefault(); changeView('chat'); }}><span className="app-logo"><Icon name="sparkles" size={24}/></span><span>本地知库</span></a>
        <button className="icon-button" aria-label="收起侧栏" onClick={() => setSidebarOpen(false)}><Icon name="sidebar" size={19}/></button></div>
      <button className="nav-item new-conversation" disabled={busy} onClick={startConversation}><Icon name="edit" size={19}/>新对话</button>
      <nav aria-label="主导航">{[['chat', '知识问答', 'chat'], ['library', '文档管理', 'library'], ['search', '原文检索', 'search']].map(([value, label, icon]) =>
        <button key={value} className={`nav-item ${view === value ? 'active' : ''}`} aria-current={view === value ? 'page' : undefined} disabled={busy} onClick={() => changeView(value)}><Icon name={icon} size={19}/>{label}</button>)}</nav>
      <section className="conversation-history" aria-label="聊天记录"><h2>聊天记录</h2>
        {conversations.filter(item => item.messages.length).map(item => <div className={`history-row ${view === 'chat' && item.id === conversation.id ? 'active' : ''}`} key={item.id}>
          <button disabled={busy} onClick={() => chooseConversation(item.id)} title={item.messages[0].question}>{item.messages[0].question}</button>
          <button className="icon-button" disabled={busy} aria-label={`删除对话：${item.messages[0].question}`} onClick={() => removeConversation(item.id)}><Icon name="trash" size={14}/></button>
        </div>)}
        {!conversations.some(item => item.messages.length) && <p className="history-empty">开始对话后会显示在这里。<br/>保存在当前浏览器中。</p>}
      </section>
      <div className="sidebar-bottom"><div className="local-badge"><span className="local-avatar">L</span><div><strong>本地工作空间</strong><span>在你的电脑上运行</span></div></div>
        {view !== 'chat' && <a href={`${__LOCAL_RAG_API_URL__}/docs`} target="_blank" rel="noreferrer" className="api-docs">API 文档<Icon name="external" size={14}/></a>}</div>
    </aside>
    <div className="main-shell">{view !== 'chat' && <header className="topbar">
      <button className="icon-button" aria-label={sidebarOpen ? '收起侧栏' : '展开侧栏'} aria-expanded={sidebarOpen} aria-controls="workspace-sidebar" onClick={() => setSidebarOpen(value => !value)}><Icon name="sidebar"/></button>
      <div className="breadcrumb"><span>工作空间</span><Icon name="chevron" size={12}/><b>{view === 'library' ? '文档管理' : '原文检索'}</b></div>
      <div className="topbar-right"><span className={`connection ${connection}`}><span/>{connection === 'online' ? 'API 已连接' : connection === 'offline' ? 'API 未连接' : '检查连接中'}</span></div></header>}
      <main className={`main-content ${view === 'chat' ? 'chat-main' : 'management-main'}`}>
        {connection === 'offline' && <div className="connection-banner" role="alert"><Icon name="info" size={19}/><div><strong>文档服务暂时无法连接</strong><p>请确认 API 服务已启动并保持运行，然后刷新连接。</p></div><button className="button secondary" onClick={() => { checkHealth(); if (view !== 'chat') refresh(); }}>重新连接</button></div>}
        <Notice notice={notice} onClose={() => setNotice(null)}/>
        <Notice notice={chatStorageError ? { type: 'error', message: chatStorageError } : null} onClose={() => setChatStorageError('')}/>
        {operation && operation.kind !== 'batch' && <div className="operation-banner" role="status"><span className="spinner"/><div><strong>正在{operationNames[operation.kind]}「{operation.filename}」</strong><span>已等待 {duration(seconds)} · 完成后会自动更新列表，请保持页面打开。</span></div></div>}
        {view === 'library' && <DocumentLibrary documents={documents.map(doc => doc.document_id === operation?.id ? { ...doc, status: operation.kind === 'delete' ? 'deleting' : 'indexing' } : doc)} health={health} loading={loading} busy={busy} onUpload={upload}
          uploadQueue={uploadQueue} batchActive={operation?.kind === 'batch'} stoppingUploads={stoppingUploads}
          onStopUploads={() => { uploadStop.current = true; setStoppingUploads(true); }}
          onRetryUploads={() => startUploads(prepareUploadRetry(uploadQueue, 'retry'))}
          onResumeUploads={() => startUploads(prepareUploadRetry(uploadQueue, 'resume'))} onClearUploads={() => setUploadQueue([])}
          onReindex={doc => mutate('reindex', doc.filename, () => api.reindex(doc.document_id), doc.document_id)} onDelete={setDeleteTarget}
          onDetail={openDetail} onRefresh={() => { checkHealth(); refresh(); }} onSearch={() => changeView('search')}/>}
        {(view === 'chat' || view === 'search') && <SearchWorkspace key={view === 'chat' ? conversation.id : view} mode={view === 'chat' ? 'ask' : 'search'} documents={view === 'search' ? documents : []} busy={!!operation} revision={revision} onSearchBusy={setSearchBusy}
          onDetail={openDetail} onLibrary={() => changeView('library')} onError={showError}
          conversation={conversation} onMessagesChange={updateMessages} sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen(value => !value)} connection={connection}/>}
        {view !== 'chat' && <footer className="page-footer"><span>本地知库 <span className="footer-separator">/</span> 让知识触手可及</span><span>React · Local RAG</span></footer>}
      </main>
    </div>
    {detail && <DocumentDetail key={`${detail.id}:${detail.initialChunk || ''}`} id={detail.id} initialChunk={detail.initialChunk} revision={revision} onClose={closeDetail}/>}
    {deleteTarget && <Modal title="删除这份文档？" onClose={closeDelete}><div className="delete-description"><span className="delete-icon"><Icon name="trash" size={25}/></span>
      <h3>{deleteTarget.filename}</h3><p>原文件、文本块和对应向量都将被删除，删除后无法继续检索这份文档。</p><code>{deleteTarget.document_id}</code></div>
      <div className="modal-actions"><button className="button secondary" onClick={closeDelete}>保留文档</button><button className="button danger" disabled={busy} onClick={() => { const doc = deleteTarget; closeDelete(); mutate('delete', doc.filename, () => api.remove(doc.document_id), doc.document_id); }}>确认删除</button></div></Modal>}
  </div>;
}
