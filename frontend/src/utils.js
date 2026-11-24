export const states = {
  ready: ['待索引', 'amber'], indexed: ['可检索', 'green'], indexing: ['索引中', 'blue'],
  failed: ['索引失败', 'red'], cleanup_required: ['待清理', 'red'], deleting: ['删除中', 'blue'],
  delete_failed: ['删除未完成', 'red'], deleted: ['已删除', 'gray'],
};

export const number = value => new Intl.NumberFormat('zh-CN').format(value || 0);
export const fileType = filename => filename.split('.').pop().toUpperCase();
export const shortId = id => id?.slice(0, 8) || '—';
export function dateTime(value) {
  if (!value) return '—';
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false,
  }).format(new Date(value));
}
export function duration(seconds) {
  return seconds < 60 ? `${seconds} 秒` : `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`;
}
