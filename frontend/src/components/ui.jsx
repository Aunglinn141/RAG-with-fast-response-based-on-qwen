import { useEffect, useId, useRef } from 'react';
import { states } from '../utils';

const paths = {
  sidebar: <><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></>,
  edit: <><path d="m15 4 5 5M4 20l5-1L21 7l-5-5L4 14z"/></>,
  chat: <path d="M20 15a3 3 0 0 1-3 3H9l-5 3V6a3 3 0 0 1 3-3h10a3 3 0 0 1 3 3z"/>,
  sparkles: <><path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5z"/><path d="m20 2 .6 1.4L22 4l-1.4.6L20 6l-.6-1.4L18 4l1.4-.6z"/></>,
  plus: <path d="M12 5v14M5 12h14"/>,
  arrowUp: <path d="M12 20V4m-6 6 6-6 6 6"/>,
  stop: <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>,
  copy: <><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V3H3v13h5"/></>,

  library: <><path d="M4 4h4v16H4zM10 4h4v16h-4zM16 5l3-1 4 15-3 1z" /></>,
  search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/></>,
  upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5"/></>,
  file: <><path d="M5 3h9l5 5v13H5zM14 3v6h5M8 13h8M8 17h5"/></>,
  refresh: <><path d="M20 7v5h-5M4 17v-5h5M5.5 8a7 7 0 0 1 12-3L20 8M4 16l2.5 3a7 7 0 0 0 12-3"/></>,
  trash: <><path d="M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7M14 10v7"/></>,
  arrow: <><path d="M5 12h14m-6-6 6 6-6 6"/></>,
  close: <path d="m6 6 12 12M6 18 18 6"/>,
  check: <path d="m5 12 4 4L19 6"/>,
  layers: <><path d="m12 3 9 5-9 5-9-5zM3 12l9 5 9-5M3 16l9 5 9-5"/></>,
  download: <><path d="M12 3v13m-5-5 5 5 5-5M4 17v4h16v-4"/></>,
  chevron: <path d="m9 5 7 7-7 7"/>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/></>,
  bolt: <path d="m13 2-9 12h7l-1 8 10-13h-7z"/>,
  external: <><path d="M14 3h7v7M21 3 11 13M10 5H4v15h15v-6"/></>,
  clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
};

export function Icon({ name, size = 20, className = '' }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={className}>{paths[name] || paths.file}</svg>;
}

export function StatusBadge({ status }) {
  const [label, tone] = states[status] || [status, 'gray'];
  return <span className={`status-badge ${tone}`}><span className="status-dot"/>{label}</span>;
}

export function Notice({ notice, onClose }) {
  if (!notice) return null;
  return <div className={`notice ${notice.type || 'info'}`} role={notice.type === 'error' ? 'alert' : 'status'}>
    <Icon name={notice.type === 'success' ? 'check' : 'info'} size={19}/>
    <span>{notice.message}</span>
    <button type="button" className="icon-button" aria-label="关闭提示" onClick={onClose}><Icon name="close" size={17}/></button>
  </div>;
}

export function EmptyState({ icon = 'file', title, children }) {
  return <div className="empty-state"><span className="empty-symbol"><Icon name={icon} size={27}/></span>
    <h3>{title}</h3><p>{children}</p></div>;
}

export function Modal({ title, onClose, children, drawer = false }) {
  const ref = useRef(null);
  const titleId = useId();
  useEffect(() => {
    const previous = document.activeElement;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    ref.current?.querySelector('button')?.focus();
    function keys(event) {
      if (event.key === 'Escape') onClose();
      if (event.key !== 'Tab') return;
      const focusable = [...(ref.current?.querySelectorAll('button:not([disabled]), a[href], input, select, textarea, [tabindex="0"]') || [])]
        .filter(node => node.getClientRects().length);
      if (!focusable.length) return;
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
    document.addEventListener('keydown', keys);
    return () => {
      document.body.style.overflow = overflow;
      document.removeEventListener('keydown', keys);
      previous?.focus();
    };
  }, [onClose]);
  return <div className={`modal-backdrop ${drawer ? 'drawer-backdrop' : ''}`} onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}>
    <section ref={ref} className={drawer ? 'detail-drawer' : 'modal'} role="dialog" aria-modal="true" aria-labelledby={titleId}>
      <div className="modal-heading"><h2 id={titleId}>{title}</h2><button className="icon-button" onClick={onClose} aria-label="关闭窗口"><Icon name="close"/></button></div>
      {children}
    </section>
  </div>;
}
