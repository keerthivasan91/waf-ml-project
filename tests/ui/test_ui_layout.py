from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "app/templates/base.html"
NAV = ROOT / "app/templates/partials/nav.html"
CSS = ROOT / "app/static/css/main.css"
JS = ROOT / "app/static/js/main.js"
SIM = ROOT / "app/templates/simulator.html"
DASHBOARD = ROOT / "app/templates/dashboard.html"
RETRAIN = ROOT / "app/templates/retraining.html"


class TestUILayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = BASE.read_text(encoding="utf-8")
        cls.nav = NAV.read_text(encoding="utf-8")
        cls.css = CSS.read_text(encoding="utf-8")
        cls.js = JS.read_text(encoding="utf-8")
        cls.sim = SIM.read_text(encoding="utf-8")
        cls.dashboard = DASHBOARD.read_text(encoding="utf-8")
        cls.retraining = RETRAIN.read_text(encoding="utf-8")

    def test_single_global_header(self):
        self.assertEqual(self.base.count('class="global-header"'), 1)

    def test_sidebar_toggle_markup(self):
        self.assertIn('id="sidebarToggle"', self.nav)
        self.assertIn('aria-label="Collapse sidebar"', self.nav)
        self.assertIn('aria-expanded="true"', self.nav)

    def test_sidebar_has_no_duplicate_brand(self):
        self.assertNotIn('class="brand-block"', self.nav)

    def test_sidebar_persistence_contract(self):
        self.assertIn('localStorage.getItem("waf-sidebar")', self.base)
        self.assertIn('localStorage.getItem(sidebarKey)', self.js)
        self.assertIn('localStorage.setItem(sidebarKey,root.dataset.sidebar)', self.js)

    def test_sidebar_toggle_behavior_contract(self):
        self.assertIn('root.dataset.sidebar=collapsed?"expanded":"collapsed"', self.js)
        self.assertIn('button.addEventListener("click"', self.js)
        self.assertIn('sidebarToggle.addEventListener("click"', self.js)
        self.assertIn('sidebarToggle.setAttribute("aria-expanded"', self.js)
        self.assertIn('sidebarIcon.textContent=collapsed?"›":"‹"', self.js)

    def test_collapsed_sidebar_geometry(self):
        self.assertIn('html[data-sidebar="collapsed"] .sidebar{width:76px', self.css)
        self.assertIn('html[data-sidebar="collapsed"] .app-content{margin-left:76px}', self.css)
        self.assertIn('.app-content{transition:margin-left .22s ease}', self.css)

    def test_collapsed_sidebar_hides_text_but_keeps_glyphs(self):
        self.assertIn('html[data-sidebar="collapsed"] .side-link>span:not(.nav-glyph):not(.side-link-dot):not(.side-link-badge){display:none}', self.css)
        self.assertIn('html[data-sidebar="collapsed"] .nav-glyph{width:34px;height:34px}', self.css)

    def test_mobile_does_not_break_collapsed_layout(self):
        self.assertIn('@media(max-width:760px){', self.css)
        self.assertIn('html[data-sidebar="collapsed"] .sidebar{width:100%', self.css)
        self.assertIn('html[data-sidebar="collapsed"] .app-content{margin-left:0}', self.css)

    def test_theme_toggle_still_present(self):
        self.assertIn('id="themeToggle"', self.base)
        self.assertIn('const key="waf-theme"', self.js)
        self.assertIn('localStorage.setItem(key,root.dataset.theme)', self.js)

    def test_admin_controls_are_present(self):
        self.assertIn('Admin Control Center', self.dashboard)
        self.assertIn('Complete Reviews', self.dashboard)
        self.assertIn('Trigger Health Audit', self.dashboard)
        self.assertIn('Prepare Retraining', self.dashboard)
        self.assertIn('/api/health/trigger-audit?error_rate=1.0', self.dashboard)
        self.assertIn('/api/feedback/trigger-retrain', self.dashboard)

    def test_retraining_control_page_exists(self):
        self.assertIn('Retraining Control', self.retraining)
        self.assertIn('Review Queue', self.retraining)
        self.assertIn('Prepare Retraining Batch', self.retraining)
        self.assertIn('Local PyTorch training is ready to start', self.retraining)
        self.assertIn('Start Local Retraining', self.retraining)
        self.assertIn('/api/feedback/trigger-retrain', self.retraining)
        self.assertIn('/api/health/trigger-audit?error_rate=1.0', self.retraining)
        self.assertIn('/api/models/reload', self.retraining)

    def test_simulator_overlap_guard_present(self):
        self.assertIn('Layout hardening: prevent score card and metric tiles from overlapping', self.sim)
        self.assertIn('.result-layout>div{min-width:0}', self.sim)
        self.assertIn('.metric-grid{grid-template-columns:repeat(4,minmax(0,1fr))}', self.sim)
        self.assertIn('@media(max-width:1200px)', self.sim)


if __name__ == "__main__":
    unittest.main()
