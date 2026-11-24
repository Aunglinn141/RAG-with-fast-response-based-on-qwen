import assert from 'node:assert/strict';
import test from 'node:test';
import { buildHistory, loadConversations, saveConversations, CONVERSATIONS_KEY, publicAnswer } from '../src/conversations.js';

const message = (id, changes = {}) => ({ id, question: `问题${id}`, answer: `回答${id}`, status: 'answered', model: 'qwen3:1.7b', ...changes });

test('legacy internal artifact IDs disappear from saved answers and follow-up history', () => {
  const internal = 'WW-FCDEB84CC7464E1F83EEE7A8295C900E';
  const answer = `${internal}是一件饰件，正式编号 WW-1234 / TEST-001。`;
  const expected = '该文物是一件饰件，正式编号 WW-1234 / TEST-001。';
  const saved = storage();
  saved.setItem(CONVERSATIONS_KEY, JSON.stringify({ version: 1, activeId: 'a',
    conversations: [{ id: 'a', messages: [message('1', { answer })] }] }));
  const loaded = loadConversations(saved);
  assert.equal(loaded.conversations[0].messages[0].answer, expected);
  assert.equal(buildHistory([message('1', { answer })])[0].answer, expected);
  assert.equal(publicAnswer(answer), expected);
  assert.equal(publicAnswer(`About ${internal.toLowerCase()}.`), 'About the artifact.');
  saveConversations(loaded.conversations, 'a', saved);
  assert.ok(!saved.getItem(CONVERSATIONS_KEY).includes(internal));
});
function storage() {
  const items = new Map();
  return { getItem: key => items.get(key) ?? null, setItem: (key, value) => items.set(key, value) };
}

test('context contains only completed question-answer pairs from the supplied conversation', () => {
  const messages = [message('1', { sources: [{ text: 'PRIVATE' }] }),
    message('2', { status: 'error' }), message('3', { status: 'pending' }),
    message('4', { status: 'cancelled' }), message('5', { status: 'insufficient_evidence' })];
  assert.deepEqual(buildHistory(messages), [
    { question: '问题1', answer: '回答1' }, { question: '问题5', answer: '回答5' },
  ]);
  assert.deepEqual(buildHistory([]), []);
  assert.deepEqual(buildHistory([message('other')]), [{ question: '问题other', answer: '回答other' }]);
});

test('history is recent, chronological and bounded by turns and Unicode characters', () => {
  const turns = Array.from({ length: 20 }, (_, i) => message(String(i)));
  const history = buildHistory(turns);
  assert.equal(history.length, 12);
  assert.equal(history[0].question, '问题8');
  assert.equal(history.at(-1).question, '问题19');
  const long = buildHistory(turns.map(turn => ({ ...turn, question: '问'.repeat(2000), answer: '📚'.repeat(3000) })));
  assert.equal(long.length, 4);
  assert.equal(Array.from(long.at(-1).answer).length, 3000);
  assert.ok(long.every(turn => !/\uFFFD/.test(turn.answer)));
});

test('refresh restores conversations and selected chat without retaining retrieved sources', () => {
  const saved = storage();
  const chats = [{ id: 'a', messages: [message('1', { sources: [{ text: 'PRIVATE' }], evidence: ['PRIVATE'] })] },
    { id: 'b', messages: [message('2')] }];
  assert.equal(saveConversations(chats, 'b', saved), '');
  assert.equal(saved.getItem(CONVERSATIONS_KEY).includes('PRIVATE'), false);
  const loaded = loadConversations(saved);
  assert.equal(loaded.activeId, 'b');
  assert.equal(loaded.conversations.length, 2);
  assert.deepEqual(buildHistory(loaded.conversations[1].messages), [{ question: '问题2', answer: '回答2' }]);
  assert.equal(chats[0].messages[0].sources[0].text, 'PRIVATE');
});

test('interrupted streams restore as stopped and cannot become memory', () => {
  const saved = storage();
  saveConversations([{ id: 'a', messages: [message('1'), message('2', { status: 'pending', answer: 'unfinished' })] }], 'a', saved);
  const messages = loadConversations(saved).conversations[0].messages;
  assert.equal(messages[1].status, 'cancelled');
  assert.equal(messages[1].answer, '');
  assert.match(messages[1].error, /未完成/);
  assert.equal(buildHistory(messages).length, 1);
});

test('deleted conversations stay deleted after reload', () => {
  const saved = storage();
  saveConversations([{ id: 'a', messages: [message('1')] }, { id: 'b', messages: [message('2')] }], 'a', saved);
  const { conversations } = loadConversations(saved);
  saveConversations(conversations.filter(chat => chat.id !== 'a'), 'b', saved);
  assert.deepEqual(loadConversations(saved).conversations.map(chat => chat.id), ['b']);
});

test('unavailable, corrupt or full browser storage has an explicit recoverable result', () => {
  const saved = storage();
  saved.setItem(CONVERSATIONS_KEY, '{broken');
  assert.match(loadConversations(saved).error, /无法读取/);
  const blocked = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('quota'); } };
  assert.match(loadConversations(blocked).error, /无法读取/);
  assert.match(saveConversations([], null, blocked), /无法保存/);
});

test('restore ignores invalid messages, duplicate identifiers and stale active selection', () => {
  const saved = storage();
  saved.setItem(CONVERSATIONS_KEY, JSON.stringify({ version: 1, activeId: 'missing', conversations: [
    { id: 'a', messages: [message('1'), message('1'), null, message('bad', { status: 'system' })] },
    { id: 'a', messages: [] }, null,
  ] }));
  const loaded = loadConversations(saved);
  assert.equal(loaded.activeId, 'a');
  assert.equal(loaded.conversations.length, 1);
  assert.deepEqual(loaded.conversations[0].messages.map(turn => turn.id), ['1']);
});
