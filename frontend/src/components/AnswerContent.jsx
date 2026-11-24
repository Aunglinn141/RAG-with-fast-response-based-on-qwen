import { Fragment } from 'react';

function inline(text) {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, index) =>
    part.startsWith('**') && part.endsWith('**') ? <strong key={index}>{part.slice(2, -2)}</strong>
      : part.startsWith('`') && part.endsWith('`') ? <code key={index}>{part.slice(1, -1)}</code>
        : <Fragment key={index}>{part}</Fragment>);
}

// React escapes every text node; no model-produced HTML is executed.
export default function AnswerContent({ text }) {
  const lines = text.split('\n'), blocks = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (line.startsWith('```')) {
      const language = line.slice(3).trim(), code = [];
      index++;
      while (index < lines.length && !lines[index].startsWith('```')) code.push(lines[index++]);
      index++;
      blocks.push(<div className="answer-code" key={blocks.length}>{language && <div>{language}</div>}<pre><code>{code.join('\n')}</code></pre></div>);
    } else if (!line.trim()) index++;
    else if (/^#{1,4}\s/.test(line)) {
      blocks.push(<h3 key={blocks.length}>{inline(line.replace(/^#{1,4}\s+/, ''))}</h3>); index++;
    } else if (/^\s*(?:[-*]|\d+[.)、])\s+/.test(line)) {
      const ordered = /^\s*\d/.test(line), list = [];
      const pattern = ordered ? /^\s*\d+[.)、]\s+/ : /^\s*[-*]\s+/;
      while (index < lines.length && pattern.test(lines[index])) list.push(<li key={index}>{inline(lines[index++].replace(pattern, ''))}</li>);
      blocks.push(ordered ? <ol key={blocks.length}>{list}</ol> : <ul key={blocks.length}>{list}</ul>);
    } else {
      const paragraph = [line]; index++;
      while (index < lines.length && lines[index].trim() && !/^(?:```|#{1,4}\s|\s*(?:[-*]|\d+[.)、])\s+)/.test(lines[index])) paragraph.push(lines[index++]);
      blocks.push(<p key={blocks.length}>{inline(paragraph.join('\n'))}</p>);
    }
  }
  return <div className="answer-content">{blocks}</div>;
}
