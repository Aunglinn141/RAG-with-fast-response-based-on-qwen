import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import ChatWorkspace from './ChatWorkspace';
import StreamingAnswer from './StreamingAnswer';
import { number, shortId } from '../utils';
import { EmptyState, Icon } from './ui';

function SourceCard({ source, rank, onDetail }) {
  const metadata = source.metadata;
  return <article className="result-card">
    <div className="result-topline">
      <span className="result-rank">{String(rank + 1).padStart(2, '0')}</span>
      <span className="result-file"><Icon name="file" size={15}/>{metadata.filename}</span>
      {Number.isFinite(source.score) && <span className="similarity" title="余弦相似度，分数越高通常越相关，不代表答案正确的概率">相似度 {source.score.toFixed(3)}</span>}
    </div>
    <p className="result-text">{source.text}</p>
    <div className="result-footer">
      <span>分块 {metadata.chunk_index}{metadata.page_number ? ` · 第 ${metadata.page_number} 页` : ''}{metadata.section_title ? ` · ${metadata.section_title}` : ''}<span className="source-document-id">文档 {shortId(metadata.document_id)}</span></span>
      <button type="button" className="text-link" onClick={() => onDetail(metadata.document_id, metadata.chunk_index)}>查看来源<Icon name="arrow" size={14}/></button>
    </div>
  </article>;
}

function Answer({ result }) {
  const insufficient = result.status === 'insufficient_evidence';
  return <section className={`answer-panel ${insufficient ? 'answer-insufficient' : ''}`} aria-label="模型回答">
    <p className="answer-text">{result.answer}</p>
  </section>;
}

const MODEL_PREFERENCE = 'local-rag.chat-model';
function savedModel() {
  try { return localStorage.getItem(MODEL_PREFERENCE) || ''; } catch { return ''; }
}

function SourceWorkspace({ mode = 'ask', documents = [], busy, revision, onSearchBusy, onDetail, onLibrary, onError }) {
  const [model, setModel] = useState(savedModel);
  const [catalog, setCatalog] = useState(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState('');
  const [modelsRevision, setModelsRevision] = useState(0);
  const [query, setQuery] = useState('');
  const [documentId, setDocumentId] = useState('');
  const [topK, setTopK] = useState(4);
  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const requestSequence = useRef(0);
  const activeRequest = useRef(null);
  const [draft, setDraft] = useState('');
  const [streamStage, setStreamStage] = useState('waiting');
  const indexed = documents.filter(doc => doc.status === 'indexed');
  const selectedUnavailable = mode === 'search' && !!documentId && !indexed.some(doc => doc.document_id === documentId);
  const indexSignature = mode === 'search' ? indexed.map(doc => `${doc.document_id}:${doc.generation}:${doc.model_version}`).join('|') : '';
  const answerMode = mode === 'ask';
  const selectedModel = catalog?.models.find(item => item.name === model);
  const modelUnavailable = answerMode && (modelsLoading || !!modelsError || !selectedModel);
  const queryLimit = answerMode ? 2000 : 4000;
  const queryTooLong = query.trim().length > queryLimit;
  const sources = answerMode ? [] : (result?.matches ?? []);
  const excluded = answerMode ? 0 : result?.excluded_documents;
  const noSearchDocuments = !answerMode && !indexed.length;

  useEffect(() => { requestSequence.current += 1; setResult(null); setDraft(''); }, [revision, indexSignature]);
  useEffect(() => () => { requestSequence.current += 1; activeRequest.current?.abort(); }, []);
  useEffect(() => {
    if (!searching) return;
    const started = performance.now();
    setElapsed(0);
    const timer = setInterval(() => setElapsed((performance.now() - started) / 1000), 1000);
    return () => clearInterval(timer);
  }, [searching]);
  useEffect(() => {
    if (!answerMode) return;
    let active = true;
    setModelsLoading(true);
    setModelsError('');
    api.models().then(data => {
      if (!active) return;
      setCatalog(data);
      setModel(current => current || data.models.find(item =>
        item.name === data.default_model || item.name === data.default_model + ':latest')?.name || '');
    }).catch(error => {
      if (active) setModelsError(error.code === 'not_found'
        ? '模型列表接口尚未加载，请重启更新后的 API，再刷新列表。' : error.message);
    }).finally(() => { if (active) setModelsLoading(false); });
    return () => { active = false; };
  }, [modelsRevision, answerMode]);
  function changeModel(value) {
    setModel(value);
    clearResult();
    try { localStorage.setItem(MODEL_PREFERENCE, value); } catch { /* Storage may be disabled. */ }
  }
  function clearResult() { requestSequence.current += 1; setResult(null); setDraft(''); setStreamStage('waiting'); }
  function editQuery(value) { setQuery(value); clearResult(); }
  async function submit(event) {
    event?.preventDefault();
    if (!query.trim() || queryTooLong || busy || searching || modelUnavailable || selectedUnavailable || noSearchDocuments) return;
    setSearching(true); onSearchBusy(true); setResult(null); setDraft(''); setStreamStage('waiting');
    const controller = new AbortController();
    activeRequest.current = controller;
    const sequence = ++requestSequence.current;
    const started = performance.now();
    const options = { top_k: Number(topK), ...(documentId ? { document_id: documentId } : {}) };
    try {
      const data = answerMode ? await api.askStream({ question: query.trim(), model: selectedModel.name }, {
        signal: controller.signal,
        onEvent: event => {
          if (sequence !== requestSequence.current) return;
          if (event.type === 'delta') setDraft(value => value + event.data.text);
          if (event.type === 'reset') setDraft('');
          if (event.type === 'stage') setStreamStage(event.data.stage);
        },
      })
        : await api.search({ query: query.trim(), ...options }, { signal: controller.signal });
      if (sequence === requestSequence.current) { setResult(data); setDraft(''); setElapsed((performance.now() - started) / 1000); }
    } catch (error) {
      if (sequence === requestSequence.current) {
        setDraft(''); setResult(null);
        if (error.code === 'request_cancelled') setStreamStage('cancelled');
        else onError(error);
      }
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null; setSearching(false); onSearchBusy(false);
      }
    }
  }
  return <>
    <div className="page-heading"><div><p className="eyebrow">{answerMode ? 'KNOWLEDGE Q&A' : 'DOCUMENT SEARCH'}</p><h1>{answerMode ? '知识问答' : '原文检索'}<span className="heading-dot">.</span></h1><p className="page-description">{answerMode ? '直接提出问题，让本地模型为你整理回答。' : '查看检索文本与来源，检查文档索引效果。'}</p></div><span className="quiet-pill"><Icon name="bolt" size={15}/>{answerMode ? '本地回答' : '检索调试'}</span></div>
    <div className={`search-layout ${answerMode ? 'question-layout' : ''}`}><div className="search-main">
      <form className="search-composer" onSubmit={submit}>
        {answerMode && <div className="model-picker">
          <label htmlFor="chat-model">回答模型
            <select id="chat-model" value={model} disabled={searching || modelsLoading}
              onChange={event => changeModel(event.target.value)} aria-describedby="model-picker-status">
              <option value="">选择项目中的本地模型</option>
              {model && !selectedModel && <option value={model} disabled>{model}（暂不可用）</option>}
              {catalog?.models.map(item => <option key={item.name} value={item.name}>
                {item.name}{item.size_bytes ? ` · ${(item.size_bytes / 1024 ** 3).toFixed(1)} GiB` : ''}
              </option>)}
            </select>
          </label>
          <button type="button" className="button secondary" disabled={searching || modelsLoading}
            onClick={() => { clearResult(); setModelsRevision(value => value + 1); }}>刷新模型</button>
          <p id="model-picker-status" role={modelsError ? 'alert' : 'status'}>
            {modelsLoading ? '正在读取项目模型…' : modelsError || (!catalog?.models.length
              ? '项目中没有可用的聊天模型，下载模型后刷新列表。'
              : !selectedModel ? '请选择可用模型；原选择可能已被删除。'
              : '使用项目中的本地模型回答，切换后从下一次提问生效。')}
          </p>
        </div>}
        <label htmlFor="knowledge-query">{answerMode ? '有什么想了解的？' : '想在文档中找到什么？'}</label>
        <textarea id="knowledge-query" value={query} onChange={event => editQuery(event.target.value)} maxLength={queryLimit} readOnly={searching} aria-busy={searching} aria-invalid={queryTooLong} aria-describedby="knowledge-query-hint"
          placeholder={answerMode ? "输入你的问题，例如：这个问题有哪些解决办法？" : "输入要查找的内容"} onKeyDown={event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') submit(event); }}/>
        <div className={`query-hint ${queryTooLong ? 'query-too-long' : ''}`} id="knowledge-query-hint"><span>{queryTooLong ? `请将问题缩短到 ${queryLimit} 个字符以内。` : answerMode ? '结合相关信息回答，无法确定时会明确说明。' : '查看相关文本块，不生成答案。'}</span><span>{query.length} / {queryLimit}</span></div>
        <div className="search-controls">{!answerMode && <div className="search-selects"><label>检索范围<select disabled={searching} value={documentId} onChange={event => { setDocumentId(event.target.value); clearResult(); }}><option value="">全部已索引文档</option>{selectedUnavailable && <option value={documentId} disabled>原文档暂不可用，请重新选择</option>}{indexed.map(doc => <option key={doc.document_id} value={doc.document_id}>{doc.filename} · {shortId(doc.document_id)}</option>)}</select></label>
          <label>检索数量<select disabled={searching} value={topK} onChange={event => { setTopK(Number(event.target.value)); clearResult(); }}>{[3,4,5,10,20].map(n => <option value={n} key={n}>{n} 个文本块</option>)}</select></label></div>}
          {searching && <button type="button" className="button secondary" onClick={() => activeRequest.current?.abort()}>{answerMode ? '停止回答' : '停止检索'}</button>}
          <button className="button primary" type="submit" disabled={!query.trim() || queryTooLong || busy || searching || modelUnavailable || selectedUnavailable || noSearchDocuments}>{searching ? <span className="spinner"/> : <Icon name={answerMode ? 'arrow' : 'search'} size={17}/>} {searching ? (answerMode ? '正在回答' : '检索中') : (answerMode ? '提问' : '开始检索')}</button></div>
      </form>
      {noSearchDocuments && <div className="inline-warning">还没有可检索的文档。<button type="button" className="text-link" onClick={onLibrary}>前往文档库建立索引<Icon name="arrow" size={14}/></button></div>}
      {selectedUnavailable && <div className="inline-warning">原来选择的文档已不在可检索范围内，请重新选择文档。</div>}
      {!searching && streamStage === 'cancelled' && <p className="stream-status" role="status">{answerMode ? '已停止本次回答，未完成的草稿已清除。' : '已停止本次检索。'}</p>}
      {searching && answerMode ? <StreamingAnswer text={draft} stage={streamStage} elapsed={elapsed}/> : searching ? <div className="search-loading" role="status"><span className="spinner"/><h3>等待检索结果</h3><p>已等待 {elapsed.toFixed(0)} 秒</p><p>服务繁忙时请求会排队，你可以随时停止检索。</p></div> : result ? <section className="search-results" aria-live="polite">
        <div className="results-heading"><h2>{answerMode ? '本次回答' : '检索结果'}{!answerMode && <span> {sources.length}</span>}</h2><span>用时 {elapsed.toFixed(2)} 秒</span></div>
        {excluded > 0 && <div className="inline-warning">有 {excluded} 份文档的模型版本不同，暂未参与检索，请重新索引。</div>}
        {answerMode && <Answer result={result}/>}
        {!sources.length && !answerMode && <EmptyState icon="search" title="暂无检索结果">检查所选文档是否已完成索引，或尝试调整检索范围。</EmptyState>}
        {!answerMode && sources.map((source, index) => <SourceCard key={source.chunk_id} source={source} rank={index} onDetail={onDetail}/>)}
      </section> : <div className="search-welcome"><div className="search-orbit"><Icon name={answerMode ? 'file' : 'search'} size={34}/></div><h2>{answerMode ? '提出问题，获取回答' : '从一个问题，找到一段原文'}</h2><p>{answerMode ? '系统会自动查找相关信息，你只需要描述想了解的问题。' : '搜索会理解问题的含义，在已索引的文档中查找相关文本。'}</p><div className="search-suggestions"><span>试着问问</span>{(answerMode ? ['这件事的来龙去脉是什么？','解决这个问题有哪些步骤？'] : ['文档中提到了哪些关键步骤？','这份资料对这个问题有哪些说明？']).map(value => <button type="button" key={value} onClick={() => editQuery(value)}>{value}<Icon name="arrow" size={14}/></button>)}</div></div>}
    </div>{!answerMode && <aside className="search-aside"><div className="aside-symbol"><Icon name="layers" size={25}/></div><p className="eyebrow">YOUR KNOWLEDGE BASE</p><h2>有依据的回答，<br/>从你的文档开始。</h2>
      <div className="aside-stat"><strong>{indexed.length}</strong><span>份文档可供检索</span></div><div className="aside-stat"><strong>{number(indexed.reduce((sum, doc) => sum + doc.chunk_count, 0))}</strong><span>个文本块已建立索引</span></div>
      <div className="aside-divider"/><h3>让问题更具体</h3><p>描述人物、主题或场景，并选择相关文档，有助于找到需要的资料。</p><p>{answerMode ? '输入具体问题，查看本地模型生成的回答。资料不足时会明确说明。' : '当前只检索原文。切换到「知识问答」，可根据检索内容生成回答。'}</p>
      <button type="button" className="text-link" onClick={onLibrary}>管理文档库<Icon name="arrow" size={15}/></button></aside>}</div>
  </>;
}

export default function SearchWorkspace(props) {
  return props.mode === 'search' ? <SourceWorkspace {...props}/> : <ChatWorkspace {...props}/>;
}
