const stages = {
  waiting: '等待服务响应',
  retrieving: '正在查找相关信息',
  generating: '正在生成回答',
  retrying: '正在重新生成简洁回答',
  verifying: '正在整理回答',
};

export default function StreamingAnswer({ text, stage, elapsed }) {
  return <section className="search-results streaming-results" aria-label="正在生成的回答" aria-busy="true">
    <div className="results-heading"><h2>{stages[stage] || stages.generating}</h2><span>已等待 {elapsed.toFixed(0)} 秒</span></div>
    {text ? <div className="answer-panel">
      <p className="answer-text">{text}<span className="stream-cursor" aria-hidden="true"/></p>
      <p className="stream-status" role="status">回答生成中，内容尚未完整。</p>
    </div> : <div className="search-loading" role="status"><span className="spinner"/>
      <p>{stage === 'waiting' ? '服务繁忙时请求会排队，你可以随时停止回答。' : '首次加载模型可能需要一些时间，生成后会逐步显示回答。'}</p></div>}
  </section>;
}
