// HTTP smoke check only: validates module delivery, not browser execution or UI.
const entry = new URL(process.argv[2] || 'http://127.0.0.1:5173');
if (entry.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(entry.hostname)
    || entry.username || entry.password) {
  throw new Error('Use a local HTTP URL, for example http://127.0.0.1:5173');
}

const pending = [];
const visited = new Set();
const failures = [];
let modules = 0;
let styles = 0;

function label(url) {
  return url.pathname + url.search;
}

// Tokenize enough syntax to find imports without matching quoted diagnostic
// messages, comments, or CSS strings emitted by Vite. This is not a JS parser.
function imports(source) {
  const tokens = [...source.matchAll(/\/\/[^\n]*|\/\*[\s\S]*?\*\/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|[A-Za-z_$][\w$]*|[^\s]/g)]
    .map(match => match[0]).filter(token => !token.startsWith('//') && !token.startsWith('/*'));
  const isString = token => token?.startsWith('"') || token?.startsWith("'");
  const result = [];
  for (let i = 0; i < tokens.length; i++) {
    const keyword = tokens[i];
    if (!['import', 'export'].includes(keyword) || tokens[i - 1] === '.') continue;
    let next = i + 1;
    if (keyword === 'import' && isString(tokens[next])) {
      result.push(tokens[next].slice(1, -1));
      continue;
    }
    if (keyword === 'import' && tokens[next] === '(') {
      if (isString(tokens[next + 1]) && [')', ','].includes(tokens[next + 2])) {
        result.push(tokens[next + 1].slice(1, -1));
      }
      continue;
    }
    // A default binding can precede a named/namespace import.
    if (keyword === 'import' && /^[A-Za-z_$][\w$]*$/.test(tokens[next] || '')) {
      next++;
      if (tokens[next] === ',') next++;
    }
    if (tokens[next] === '{') {
      while (next < tokens.length && tokens[next] !== '}') next++;
      next++;
    } else if (tokens[next] === '*') {
      next++;
      if (tokens[next] === 'as') next += 2;
    }
    if (tokens[next] === 'from' && isString(tokens[next + 1])) {
      result.push(tokens[next + 1].slice(1, -1));
    }
  }
  return result;
}

function enqueue(specifier, importer) {
  if (!/^(?:\.{1,2}\/|\/|https?:\/\/)/.test(specifier)) {
    failures.push(`${label(importer)}: unresolved bare or unsupported import ${JSON.stringify(specifier)}`);
    return;
  }
  const url = new URL(specifier, importer);
  url.hash = '';
  if (url.origin !== entry.origin) {
    failures.push(`${label(importer)}: external module ${url.href}; this check follows local modules only`);
    return;
  }
  if (!visited.has(url.href)) {
    visited.add(url.href);
    pending.push({ url, importer });
  }
}

async function read(url, importer) {
  try {
    const response = await fetch(url, {
      signal: AbortSignal.timeout(10000), redirect: 'manual', cache: 'no-store',
    });
    if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
    return {
      source: await response.text(),
      type: (response.headers.get('content-type') || '').split(';')[0].trim().toLowerCase(),
    };
  } catch (error) {
    failures.push(`${label(url)}: ${error.message}${importer ? ` (imported by ${label(importer)})` : ''}`);
    return null;
  }
}

console.log(`Checking frontend module delivery at ${entry.href}`);
const page = await read(entry);
if (page && page.type !== 'text/html') failures.push(`${label(entry)}: expected text/html, received ${page.type || '(missing Content-Type)'}`);
if (page?.type === 'text/html') {
  let scripts = 0;
  for (const match of page.source.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)) {
    const attributes = Object.fromEntries([...match[1].matchAll(/([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/g)]
      .map(attribute => [attribute[1].toLowerCase(), attribute[2] ?? attribute[3] ?? attribute[4]]));
    if (attributes.type?.toLowerCase() !== 'module') continue;
    scripts++;
    if (attributes.src) enqueue(new URL(attributes.src.replaceAll('&amp;', '&'), entry).href, entry);
    else for (const specifier of imports(match[2])) enqueue(specifier, entry);
  }
  if (!scripts) failures.push(`${label(entry)}: no module scripts found`);
}

while (pending.length) {
  if (visited.size > 200) {
    failures.push('Module graph exceeded 200 URLs; check for repeated changing dependency URLs.');
    break;
  }
  const { url, importer } = pending.shift();
  const resource = await read(url, importer);
  if (!resource) continue;
  if (resource.type === 'text/css' && url.pathname.endsWith('.css')) {
    styles++;
    continue;
  }
  if (!/^(?:text|application)\/(?:javascript|ecmascript|x-javascript)$/.test(resource.type)) {
    failures.push(`${label(url)}: expected JavaScript${url.pathname.endsWith('.css') ? ' or CSS' : ''}, received ${resource.type || '(missing Content-Type)'} (imported by ${label(importer)})`);
    continue;
  }
  modules++;
  for (const specifier of imports(resource.source)) enqueue(specifier, url);
}

if (failures.length) {
  console.error(`FAILED: ${failures.length} module delivery problem(s):`);
  for (const failure of failures) console.error(`- ${failure}`);
  console.error('A successful homepage response alone does not mean React can load. Check the frontend terminal and restart the dev server after fixing its dependency cache.');
  process.exitCode = 1;
} else {
  console.log(`PASS: HTML and ${modules} JavaScript modules${styles ? ` plus ${styles} CSS files` : ''} loaded successfully. Browser rendering and interactions are not tested.`);
}
