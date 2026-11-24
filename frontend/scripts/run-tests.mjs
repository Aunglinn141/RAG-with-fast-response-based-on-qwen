// Run client and rendered-workspace checks with the same supported Node versions as Vite.
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const frontend = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const [major, minor] = process.versions.node.split('.').map(Number);
const compatible = major === 20 ? minor >= 19 : major === 22 ? minor >= 12 : major > 22;
const executable = compatible ? process.execPath
  : resolve(frontend, '..', '.runtime', 'node', process.platform === 'win32' ? 'node.exe' : 'node');
if (!existsSync(executable)) {
  console.error('Frontend tests require Node 20.19+ or 22.12+. Install a supported Node version.');
  process.exit(1);
}
const child = spawn(executable, ['--test', 'tests/api.test.mjs', 'tests/streaming.test.mjs', 'tests/workspace.test.mjs', 'tests/upload-queue.test.mjs', 'tests/conversations.test.mjs'], {
  cwd: frontend, stdio: 'inherit', windowsHide: true,
});
child.on('error', error => { console.error(error.message); process.exitCode = 1; });
child.on('exit', code => { process.exitCode = code ?? 1; });
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
