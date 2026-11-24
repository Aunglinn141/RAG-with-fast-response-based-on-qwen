import { summarizeUploads } from '../upload-queue';
import { Icon } from './ui';

const labels = { queued: '等待中', processing: '处理中', indexed: '已索引', saved: '待索引',
  invalid: '未通过检查', failed: '失败', uncertain: '待核对', paused: '未导入' };

export default function UploadQueue({ items, active, stopping, onStop, onRetry, onResume, onClear, onDetail }) {
  if (!items.length) return null;
  const count = summarizeUploads(items);
  const current = items.find(item => item.status === 'processing');
  return <section className="upload-queue" aria-labelledby="upload-queue-title">
    <div className="queue-heading"><div><h2 id="upload-queue-title">{active ? '正在批量导入' : count.remaining ? '导入已暂停' : '本批导入结果'}</h2>
      <p role="status">{active && current ? `正在处理：${current.filename}` : `已索引 ${count.indexed} · 待索引 ${count.saved} · 失败 ${count.failed} · 待核对 ${count.uncertain} · 未导入 ${count.remaining}`}</p></div>
      <div className="queue-actions">{active ? <button className="button secondary" disabled={stopping} onClick={onStop}>{stopping ? '当前文件完成后停止' : '停止后续导入'}</button> : <>
        {items.some(item => item.status === 'paused') && <button className="button secondary" onClick={onResume}>继续未导入项</button>}
        {items.some(item => item.status === 'failed') && <button className="button secondary" onClick={onRetry}>重试失败项</button>}
        <button className="icon-button" onClick={onClear} aria-label="清除导入记录" title="清除导入记录"><Icon name="close" size={18}/></button>
      </>}</div></div>
    <div className="queue-progress"><progress max={count.total} value={count.handled} aria-label="已处理文件数量"/><span>已处理 {count.handled} / {count.total}</span></div>
    <ul className="queue-files">{items.map(item => <li key={item.id} className={`queue-file queue-${item.status}`}>
      <span className="queue-file-icon">{item.status === 'processing' ? <span className="spinner"/> : <Icon name={item.status === 'indexed' ? 'check' : 'file'} size={19}/>}</span>
      <div className="queue-file-copy"><strong>{item.filename}</strong><p>{item.message || '等待前面的文件完成'}</p>
        {item.documentId && <button className="text-link" disabled={active} onClick={() => onDetail(item.documentId)}>查看已保存文档</button>}</div>
      <span className="queue-state">{labels[item.status]}</span>
    </li>)}</ul>
    <p className="queue-note">{active ? '按顺序处理文件。停止后会完成当前文件，请保持页面打开。' : '进度按已处理文件数计算。失败或待核对的文件不会被标记为成功。'}</p>
  </section>;
}
