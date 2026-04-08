/**
 * app/static/js/live_logs.js
 * Polls /api/logs/recent every 5 seconds and prepends new rows
 * to the log table without a full page refresh.
 * Relies on the partial template log_row.html rendered server-side,
 * but here we do simple client-side row building for speed.
 */

const DECISION_CLASS = { block: 'row--block', log: 'row--log', allow: '' };
const ATTACK_COLORS  = {
  sqli: 'attack-badge--sqli', xss: 'attack-badge--xss',
  lfi: 'attack-badge--lfi',   other: 'attack-badge--other',
  cmdi: 'attack-badge--cmdi', normal: 'attack-badge--normal',
  false: 'attack-badge--false',
};
const SCORE_CLASS = score =>
  score >= 70 ? 'score-pill--high' : score >= 30 ? 'score-pill--med' : 'score-pill--low';

let _seenIds = new Set();
let _paused  = false;

function badgeClass(label) {
  const key = label.split('_')[0];
  return ATTACK_COLORS[key] || 'attack-badge--other';
}

function buildRow(log) {
  const ts = log.timestamp
    ? new Date(log.timestamp).toLocaleTimeString()
    : '—';

  const tr = document.createElement('tr');
  tr.className = DECISION_CLASS[log.decision] || '';
  tr.dataset.requestId = log.request_id;
  tr.innerHTML = `
    <td class="mono nowrap">${ts}</td>
    <td class="mono dim">${log.ip || '—'}</td>
    <td class="mono">${log.method}</td>
    <td class="url-cell" title="${log.url}">${
      log.url.length > 60 ? log.url.slice(0, 60) + '…' : log.url
    }</td>
    <td><span class="attack-badge ${badgeClass(log.label)}">${log.label}</span></td>
    <td><span class="score-pill ${SCORE_CLASS(log.score)}">${log.score}</span></td>
    <td class="mono dim">${log.layer}</td>
    <td class="mono dim">${log.latency_ms}ms</td>
  `;
  return tr;
}

async function pollLogs() {
  if (_paused) return;
  try {
    const r    = await fetch('/api/logs/recent?limit=50');
    const logs = await r.json();
    const tbody = document.querySelector('#log-table tbody');
    if (!tbody) return;

    let added = 0;
    // iterate newest-first (API returns sorted desc)
    for (const log of logs) {
      if (_seenIds.has(log.request_id)) continue;
      _seenIds.add(log.request_id);
      const row = buildRow(log);
      row.style.opacity = '0';
      tbody.prepend(row);
      requestAnimationFrame(() => {
        row.style.transition = 'opacity 0.3s';
        row.style.opacity    = '1';
      });
      added++;
    }

    // Keep table from growing beyond 200 rows
    while (tbody.rows.length > 200) {
      const last = tbody.rows[tbody.rows.length - 1];
      _seenIds.delete(last.dataset.requestId);
      tbody.deleteRow(tbody.rows.length - 1);
    }

    // Update counters if present
    const blockedEl = document.getElementById('live-blocked-count');
    if (blockedEl) {
      const blocked = logs.filter(l => l.decision === 'block').length;
      blockedEl.textContent = blocked;
    }

  } catch (e) {
    console.warn('Log poll failed:', e);
  }
}

function startLiveLogs(intervalMs = 5000) {
  // Initial load
  pollLogs();
  const timer = setInterval(pollLogs, intervalMs);

  // Pause/resume button
  const btn = document.getElementById('pause-btn');
  if (btn) {
    btn.addEventListener('click', () => {
      _paused = !_paused;
      btn.textContent = _paused ? '▶ Resume' : '⏸ Pause';
      btn.style.color = _paused ? 'var(--amber)' : '';
    });
  }

  return timer;
}

document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('log-table')) {
    startLiveLogs(5000);
  }
});