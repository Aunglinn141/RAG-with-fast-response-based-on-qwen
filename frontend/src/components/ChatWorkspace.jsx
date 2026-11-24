import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { Icon } from './ui';
import StreamingAnswer from './StreamingAnswer';
import AnswerContent from './AnswerContent';
import { buildHistory, publicAnswer } from '../conversations';

const preference = 'local-rag.chat-model';
function savedModel() { try { return localStorage.getItem(preference) || ''; } catch { return ''; } }
function identifier() { return globalThis.crypto?.randomUUID?.() || String(Date.now()) + Math.random(); }

export default function ChatWorkspace({ conversation, onMessagesChange, busy = false, onSearchBusy = () => {},
  onLibrary = () => {}, onError = () => {}, onToggleSidebar = () => {}, sidebarOpen = true, connection = 'checking' }) {
  const [localMessages, setLocalMessages] = useState([]);
  const messages = conversation?.messages || localMessages;
  const updateMessages = onMessagesChange || setLocalMessages;
  const [query, setQuery] = useState('');
  const [model, setModel] = useState(savedModel);
  const [catalog, setCatalog] = useState(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState('');
  const [modelRevision, setModelRevision] = useState(0);
  const [searching, setSearching] = useState(false);
  const [draft, setDraft] = useState('');
  const [stage, setStage] = useState('waiting');
  const [elapsed, setElapsed] = useState(0);
  const [copied, setCopied] = useState('');
  const activeRequest = useRef(null), sequence = useRef(0), textarea = useRef(null), transcript = useRef(null);
  const followBottom = useRef(true);
  const selectedModel = catalog?.models.find(item => item.name === model);
  const unavailable = modelsLoading || !!modelsError || !selectedModel;
  const tooLong = query.trim().length > 2000;

  useEffect(() => {
    let active = true;
    setModelsLoading(true); setModelsError('');
    api.models().then(data => {
      if (!active) return;
      setCatalog(data);
      setModel(current => current || data.models.find(item =>
        item.name === data.default_model || item.name === data.default_model + ':latest')?.name || '');
    }).catch(error => { if (active) setModelsError(error.message); })
      .finally(() => { if (active) setModelsLoading(false); });
    return () => { active = false; };
  }, [modelRevision]);
  useEffect(() => () => { sequence.current++; activeRequest.current?.abort(); }, []);
  useEffect(() => {
    if (!searching) return;
    const started = performance.now();
    const timer = setInterval(() => setElapsed((performance.now() - started) / 1000), 1000);
    return () => clearInterval(timer);
  }, [searching]);
  useEffect(() => {
    if (!textarea.current) return;
    textarea.current.style.height = 'auto';
    textarea.current.style.height = Math.min(180, textarea.current.scrollHeight) + 'px';
  }, [query]);
  useEffect(() => {
    if (followBottom.current && transcript.current) transcript.current.scrollTop = transcript.current.scrollHeight;
  }, [messages, draft, stage]);

  function patchTurn(id, patch) {
    updateMessages(items => items.map(item => item.id === id ? { ...item, ...patch } : item));
  }
  function changeModel(value) {
    setModel(value);
    try { localStorage.setItem(preference, value); } catch { /* Private browsing can disable storage. */ }
  }
  async function submit(event) {
    event?.preventDefault();
    if (activeRequest.current || searching || busy || unavailable || tooLong || !query.trim()) return;
    const question = query.trim(), turnId = identifier(), controller = new AbortController();
    const history = buildHistory(messages);
    const request = ++sequence.current, started = performance.now();
    activeRequest.current = controller; followBottom.current = true;
    setSearching(true); onSearchBusy(true); setQuery(''); setDraft(''); setStage('waiting'); setElapsed(0);
    updateMessages(items => [...items, { id: turnId, question, model: selectedModel.name, status: 'pending', answer: '' }]);
    try {
      const result = await api.askStream({ question, model: selectedModel.name, history }, {
        signal: controller.signal,
        onEvent: event => {
          if (request !== sequence.current) return;
          if (event.type === 'delta') setDraft(value => value + event.data.text);
          if (event.type === 'reset') setDraft('');
          if (event.type === 'stage') setStage(event.data.stage);
        },
      });
      if (request !== sequence.current) return;
      patchTurn(turnId, { answer: result.answer, status: result.status, elapsed: (performance.now() - started) / 1000 });
      setDraft('');
    } catch (error) {
      if (request !== sequence.current) return;
      setDraft('');
      patchTurn(turnId, { status: error.code === 'request_cancelled' ? 'cancelled' : 'error',
        error: error.code === 'request_cancelled' ? '已停止回答，未完成的草稿已清除。' : error.message });
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null; setSearching(false); onSearchBusy(false);
        textarea.current?.focus();
      }
    }
  }
  async function copyAnswer(turn) {
    try { await navigator.clipboard.writeText(publicAnswer(turn.answer)); setCopied(turn.id); }
    catch { onError(new Error('复制未成功，可以选中回答文字后复制。')); }
  }
  return <section className={`chat-workspace ${messages.length ? 'has-messages' : 'is-empty'}`} aria-label="知识问答">
    <header className="chat-topbar">
      <button type="button" className="icon-button sidebar-toggle" aria-label={sidebarOpen ? '收起侧栏' : '展开侧栏'} aria-expanded={sidebarOpen} aria-controls="workspace-sidebar" onClick={onToggleSidebar}><Icon name="sidebar"/></button>
      <div className="chat-model-picker"><label className="visually-hidden" htmlFor="chat-model">回答模型</label>
        <select id="chat-model" value={model} disabled={searching || modelsLoading} onChange={event => changeModel(event.target.value)}>
          <option value="">选择本地模型</option>
          {model && !selectedModel && <option value={model} disabled>{model}（暂不可用）</option>}
          {catalog?.models.map(item => <option key={item.name} value={item.name}>{item.name}</option>)}
        </select><button className="icon-button" type="button" disabled={searching || modelsLoading} onClick={() => setModelRevision(value => value + 1)} aria-label="刷新模型" title="刷新模型"><Icon name="refresh" size={16}/></button>
      </div>
      <span className={`connection ${connection}`}><span/>{connection === 'online' ? '本地运行' : connection === 'offline' ? '服务未连接' : '检查连接中'}</span>
    </header>
    <div className="chat-stage">
      {!messages.length && <div className="chat-welcome"><span className="chat-wordmark"><Icon name="sparkles" size={27}/></span><h1>有什么可以帮忙的？</h1><p>从一个问题开始，把想法聊清楚。</p></div>}
      {!!messages.length && <div className="chat-transcript" ref={transcript} role="log" aria-label="当前对话"
        onScroll={event => { const node = event.currentTarget; followBottom.current = node.scrollHeight - node.scrollTop - node.clientHeight < 100; }}>
        <div className="chat-messages">{messages.map(turn => <article className="chat-turn" key={turn.id}>
          <div className="chat-user"><span className="visually-hidden">你：</span><p>{turn.question}</p></div>
          <div className="chat-assistant"><span className="assistant-mark" aria-hidden="true"><Icon name="sparkles" size={21}/></span><div className="assistant-body">
            {turn.status === 'pending' ? <StreamingAnswer text={draft} stage={stage} elapsed={elapsed}/>
              : turn.status === 'error' || turn.status === 'cancelled' ? <p className={`chat-turn-status ${turn.status}`} role={turn.status === 'error' ? 'alert' : 'status'}>{turn.error}</p>
                : <><AnswerContent text={publicAnswer(turn.answer)}/><div className="answer-actions">
                  <button type="button" className="icon-button" aria-label={copied === turn.id ? '已复制回答' : '复制回答'} title={copied === turn.id ? '已复制' : '复制回答'} onClick={() => copyAnswer(turn)}><Icon name={copied === turn.id ? 'check' : 'copy'} size={17}/></button>
                  <span>{turn.model} · {turn.elapsed?.toFixed(1)} 秒</span>
                </div></>}
          </div></div>
        </article>)}</div>
      </div>}
      <div className="chat-composer-wrap">
        {unavailable && <p className="chat-model-status" role={modelsError ? 'alert' : 'status'}>{modelsLoading ? '正在读取项目模型…' : modelsError || '请选择项目中可用的聊天模型。'}</p>}
        <form className="chat-composer" onSubmit={submit}>
          <label className="visually-hidden" htmlFor="knowledge-query">有什么想了解的？</label>
          <textarea id="knowledge-query" ref={textarea} rows={1} value={query} maxLength={2000} readOnly={searching} aria-invalid={tooLong} aria-describedby="chat-input-hint"
            onChange={event => setQuery(event.target.value)} placeholder="输入问题，开始对话"
            onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) submit(event); }}/>
          <div className="chat-composer-tools"><button type="button" className="icon-button composer-add" disabled={busy || searching} onClick={onLibrary} aria-label="导入文档" title="前往文档管理批量导入"><Icon name="plus" size={23}/></button>
            <span className="composer-mode"><Icon name="layers" size={15}/>本地问答</span>
            <span className="composer-count">{query.length > 1600 ? `${query.length} / 2000` : ''}</span>
            {searching ? <button type="button" className="chat-send" aria-label="停止回答" title="停止回答" onClick={() => activeRequest.current?.abort()}><Icon name="stop" size={17}/></button>
              : <button type="submit" className="chat-send" aria-label="提问" title="发送问题" disabled={busy || unavailable || !query.trim() || tooLong}><Icon name="arrowUp" size={22}/></button>}
          </div>
        </form>
        {!messages.length && <div className="chat-suggestions">{[['人物关系', '请梳理主要人物之间的关系。'], ['事件时间线', '请按时间顺序整理主要事件。'], ['总结观点', '请总结主要观点和结论。']].map(([label, question]) =>
          <button key={label} type="button" onClick={() => { setQuery(question); textarea.current?.focus(); }}>{label}</button>)}</div>}
        <p className="chat-input-hint" id="chat-input-hint">{tooLong ? '请将问题缩短到 2000 字符以内。' : '同一对话会记住近期上下文。Shift + Enter 换行。'}</p>
      </div>
    </div>
  </section>;
}
