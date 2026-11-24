export const CONVERSATIONS_KEY = 'local-rag.conversations.v1';
export const MAX_HISTORY_TURNS = 12;
export const MAX_HISTORY_CHARACTERS = 24000;
const completed = new Set(['answered', 'insufficient_evidence']);

// Older saved answers may contain the catalog's internal UUID. Real accession
// numbers (for example WW-1234 or TEST-001) are not matched by this format.
export function publicAnswer(answer = '') {
  return answer.replace(/WW-[0-9a-f]{32}/gi, /[\u3400-\u9fff]/u.test(answer) ? '该文物' : 'the artifact');
}

export function newConversation() {
  return { id: globalThis.crypto?.randomUUID?.() || String(Date.now()) + Math.random(), messages: [] };
}

function text(value, maximum) {
  return typeof value === 'string' ? Array.from(value.trim()).slice(0, maximum).join('') : '';
}

// Send only completed exchanges from the active conversation, without source data
// or UI state. The backend applies a second, model-specific context budget.
export function buildHistory(messages = []) {
  const history = [];
  let characters = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const turn = messages[i];
    if (!completed.has(turn.status)) continue;
    const question = text(turn.question, 2000), answer = publicAnswer(text(turn.answer, 3000));
    if (!question || !answer) continue;
    const length = Array.from(question).length + Array.from(answer).length;
    if (history.length === MAX_HISTORY_TURNS || characters + length > MAX_HISTORY_CHARACTERS) break;
    history.unshift({ question, answer });
    characters += length;
  }
  return history;
}

function savedTurn(turn) {
  if (!turn || typeof turn.id !== 'string' || !turn.id || !text(turn.question, 2000)) return null;
  if (!completed.has(turn.status) && !['error', 'cancelled', 'pending'].includes(turn.status)) return null;
  const interrupted = turn.status === 'pending';
  const status = interrupted ? 'cancelled' : turn.status;
  return { id: turn.id, question: text(turn.question, 2000), status,
    answer: completed.has(status) ? publicAnswer(text(turn.answer, 3000)) : '',
    model: text(turn.model, 256),
    ...(Number.isFinite(turn.elapsed) && turn.elapsed >= 0 ? { elapsed: turn.elapsed } : {}),
    ...(!completed.has(status) ? { error: interrupted ? '上次回答未完成，请重新提问。' : text(turn.error, 1000) || '本次回答未完成。' } : {}),
  };
}

function savedConversations(conversations) {
  const ids = new Set();
  return conversations.flatMap(chat => {
    if (!chat || typeof chat.id !== 'string' || !chat.id || ids.has(chat.id) || !Array.isArray(chat.messages)) return [];
    ids.add(chat.id);
    const messageIds = new Set();
    const messages = chat.messages.map(savedTurn).filter(turn => {
      if (!turn || messageIds.has(turn.id)) return false;
      messageIds.add(turn.id);
      return true;
    });
    return [{ id: chat.id, messages }];
  });
}

export function loadConversations(storage) {
  try {
    const raw = (storage || globalThis.localStorage)?.getItem(CONVERSATIONS_KEY);
    if (!raw) return { conversations: [], activeId: null, error: '' };
    const data = JSON.parse(raw);
    if (data?.version !== 1 || !Array.isArray(data.conversations)) throw new Error('Invalid saved conversations');
    const conversations = savedConversations(data.conversations);
    return { conversations, activeId: conversations.some(chat => chat.id === data.activeId) ? data.activeId : conversations[0]?.id || null,
      error: '' };
  } catch {
    return { conversations: [], activeId: null, error: '无法读取本机聊天记录。你可以继续本次聊天。' };
  }
}

export function saveConversations(conversations, activeId, storage) {
  try {
    const target = storage || globalThis.localStorage;
    if (!target) throw new Error('Storage unavailable');
    target.setItem(CONVERSATIONS_KEY, JSON.stringify({ version: 1, activeId,
      conversations: savedConversations(conversations) }));
    return '';
  } catch {
    return '聊天记录暂时无法保存到本机，刷新后可能丢失。请检查浏览器存储空间或删除不需要的对话。';
  }
}
