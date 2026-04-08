/* WAF-ML Dashboard — main.js */

// Auto-refresh stats section every 30s (dashboard page only)
document.addEventListener("DOMContentLoaded", () => {
  // Highlight current nav item
  const path = window.location.pathname;
  document.querySelectorAll(".nav-link").forEach(link => {
    if (link.getAttribute("href") === path) link.classList.add("active");
  });

  // Animate stat values on load
  document.querySelectorAll(".stat-value").forEach(el => {
    const raw = parseInt(el.textContent.replace(/[^0-9]/g, ""), 10);
    if (!isNaN(raw) && raw > 0) {
      let start = 0;
      const duration = 600;
      const step = raw / (duration / 16);
      const unit = el.querySelector(".stat-unit");
      const unitText = unit ? unit.outerHTML : "";
      const timer = setInterval(() => {
        start = Math.min(start + step, raw);
        el.innerHTML = Math.floor(start).toLocaleString() + unitText;
        if (start >= raw) clearInterval(timer);
      }, 16);
    }
  });

  // Animate breakdown bars
  document.querySelectorAll(".breakdown-bar").forEach(bar => {
    const target = bar.style.width;
    bar.style.width = "0";
    requestAnimationFrame(() => {
      setTimeout(() => { bar.style.width = target; }, 100);
    });
  });
});