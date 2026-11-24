// Prefer the active Node installation; an ignored project-local runtime may be
// provided for development on machines with an older system Node installation.
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve } from 'node:path';

const frontend = resolve(dirname(fileURLToPath(import.meta.url)), '..');
function compatible(version) {
  const [major, minor] = version.replace(/^v/, '').split('.').map(Number);
  return major === 20 ? minor >= 19 : major === 22 ? minor >= 12 : major > 22;
}

let executable = process.execPath;
if (!compatible(process.version)) {
  const fallback = resolve(frontend, '..', '.runtime', 'node', process.platform === 'win32' ? 'node.exe' : 'node');
  if (!existsSync(fallback)) {
    console.error('The frontend requires Node 20.19+ or 22.12+. Install a supported Node version and retry.');
    process.exit(1);
  }
  executable = fallback;
}

const cli = resolve(frontend, 'node_modules', 'vite', 'bin', 'vite.js');
if (!existsSync(cli)) {
  console.error('Frontend dependencies are missing. From the project root, run powershell -NoProfile -File ./scripts/install_frontend.ps1 first.');
  process.exit(1);
}
if (executable === process.execPath) {
  process.chdir(frontend);
  if (!process.argv.includes('--configLoader')) process.argv.push('--configLoader', 'native');
  await import(pathToFileURL(cli).href);
} else {
  const child = spawn(executable, [cli, ...process.argv.slice(2), '--configLoader', 'native'], { cwd: frontend, stdio: 'inherit' });
  child.on('error', (error) => { console.error(error.message); process.exitCode = 1; });
  child.on('exit', (code) => { process.exitCode = code ?? 1; });
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
}
