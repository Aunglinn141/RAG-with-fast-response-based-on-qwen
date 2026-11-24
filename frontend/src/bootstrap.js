// Keep this entry independent of React so a failed dependency request is visible.
const reload = document.getElementById('startup-reload');
reload.addEventListener('click', () => window.location.reload());

import('./main.jsx').catch(error => {
  console.error('GUI startup failed:', error);
  document.getElementById('startup-title').textContent = '界面加载失败';
  document.getElementById('startup-message').textContent =
    '请先按 Ctrl+F5 刷新页面。如果仍然失败，请在前端终端按 Ctrl+C，重新运行 scripts/start_frontend.ps1 后再打开页面。';
  document.getElementById('startup-error').textContent = error?.message || String(error);
  document.getElementById('startup-details').hidden = false;
  reload.hidden = false;
});
