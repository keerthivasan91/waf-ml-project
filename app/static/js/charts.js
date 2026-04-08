/**
 * app/static/js/charts.js
 * Lightweight canvas charts for the WAF dashboard.
 * No external chart library — pure Canvas 2D API.
 * Draws:  - hourly block/allow sparkline
 *         - attack-class donut
 *         - latency bar chart
 */

const WAF_COLORS = {
  red:    '#ef4444',
  amber:  '#f59e0b',
  green:  '#22c55e',
  blue:   '#3b82f6',
  purple: '#a855f7',
  teal:   '#14b8a6',
  dim:    '#383d47',
  border: '#232730',
  bg:     '#131519',
  text:   '#6b7280',
};

// ── Sparkline ─────────────────────────────────────────────────────────────────
function drawSparkline(canvasId, blocked, allowed) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = 8;

  ctx.clearRect(0, 0, W, H);

  const maxVal = Math.max(...blocked, ...allowed, 1);
  const n = blocked.length;
  const step = (W - pad * 2) / (n - 1);

  function drawLine(data, color) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    data.forEach((v, i) => {
      const x = pad + i * step;
      const y = H - pad - (v / maxVal) * (H - pad * 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Fill under line
    ctx.lineTo(pad + (n - 1) * step, H - pad);
    ctx.lineTo(pad, H - pad);
    ctx.closePath();
    ctx.fillStyle = color + '22';
    ctx.fill();
  }

  drawLine(allowed, WAF_COLORS.green);
  drawLine(blocked, WAF_COLORS.red);

  // X-axis labels (last 6 hours)
  ctx.fillStyle = WAF_COLORS.text;
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  const now = new Date();
  [5, 4, 3, 2, 1, 0].forEach((hoursAgo, i) => {
    const h = new Date(now - hoursAgo * 3600000).getHours();
    const label = String(h).padStart(2, '0') + ':00';
    const x = pad + Math.round(i * (W - pad * 2) / 5);
    ctx.fillText(label, x, H - 1);
  });
}

// ── Donut chart ───────────────────────────────────────────────────────────────
function drawDonut(canvasId, data) {
  /**
   * data = { sqli: 40, xss: 20, lfi: 15, other_attack: 25 }
   */
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2;
  const outerR = Math.min(cx, cy) - 10;
  const innerR = outerR * 0.55;

  const LABEL_COLORS = {
    sqli:         WAF_COLORS.red,
    xss:          WAF_COLORS.purple,
    lfi:          '#f97316',
    other_attack: WAF_COLORS.amber,
    cmdi:         WAF_COLORS.amber,
    normal:       WAF_COLORS.green,
  };

  const total = Object.values(data).reduce((s, v) => s + v, 0);
  if (total === 0) return;

  ctx.clearRect(0, 0, W, H);

  let angle = -Math.PI / 2;
  Object.entries(data).forEach(([label, count]) => {
    const slice = (count / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, outerR, angle, angle + slice);
    ctx.closePath();
    ctx.fillStyle = LABEL_COLORS[label] || WAF_COLORS.dim;
    ctx.fill();
    ctx.strokeStyle = WAF_COLORS.bg;
    ctx.lineWidth = 2;
    ctx.stroke();
    angle += slice;
  });

  // Hole
  ctx.beginPath();
  ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
  ctx.fillStyle = WAF_COLORS.bg;
  ctx.fill();

  // Centre text
  ctx.fillStyle = '#e8eaf0';
  ctx.font = `bold 20px Syne, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(total.toLocaleString(), cx, cy - 8);
  ctx.font = `11px JetBrains Mono, monospace`;
  ctx.fillStyle = WAF_COLORS.text;
  ctx.fillText('blocked', cx, cy + 12);
}

// ── Latency bar chart ─────────────────────────────────────────────────────────
function drawLatencyBars(canvasId, bins) {
  /**
   * bins = [{ label: '<5ms', count: 800 }, { label: '5-10ms', count: 200 }, ...]
   */
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const padL = 8, padR = 8, padT = 8, padB = 22;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;

  ctx.clearRect(0, 0, W, H);

  const maxCount = Math.max(...bins.map(b => b.count), 1);
  const barW     = chartW / bins.length;

  bins.forEach((bin, i) => {
    const barH = (bin.count / maxCount) * chartH;
    const x    = padL + i * barW + 2;
    const y    = padT + chartH - barH;
    const isHigh = bin.label.includes('>20') || bin.label.includes('20-');

    ctx.fillStyle = isHigh ? WAF_COLORS.red + 'cc' : WAF_COLORS.blue + 'cc';
    ctx.beginPath();
    ctx.roundRect(x, y, barW - 4, barH, [3, 3, 0, 0]);
    ctx.fill();

    // Label
    ctx.fillStyle = WAF_COLORS.text;
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(bin.label, x + (barW - 4) / 2, H - 4);
  });
}

// ── Auto-fetch and render on dashboard load ───────────────────────────────────
async function initCharts() {
  try {
    const res  = await fetch('/api/health/stats');
    const data = await res.json();

    // Attack breakdown donut
    if (data.attack_breakdown && Object.keys(data.attack_breakdown).length) {
      drawDonut('donut-chart', data.attack_breakdown);
    }

    // Placeholder sparkline with random data if real hourly data unavailable
    const n = 24;
    const fakeBlocked  = Array.from({length: n}, () => Math.floor(Math.random() * 50));
    const fakeAllowed  = Array.from({length: n}, () => Math.floor(Math.random() * 400 + 100));
    drawSparkline('sparkline-chart', fakeBlocked, fakeAllowed);

    // Latency bins — placeholder until real histogram endpoint added
    const latBins = [
      { label: '<1ms',   count: 650 },
      { label: '1-5ms',  count: 280 },
      { label: '5-10ms', count: 120 },
      { label: '10-20ms',count: 80  },
      { label: '>20ms',  count: 40  },
    ];
    drawLatencyBars('latency-chart', latBins);

  } catch (e) {
    console.warn('Chart data fetch failed:', e);
  }
}

document.addEventListener('DOMContentLoaded', initCharts);