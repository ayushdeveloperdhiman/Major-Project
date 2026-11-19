"""Capture dashboard screenshots (evidence for the workbook).

    python dashboard/capture_screenshots.py

Builds the static HTML, starts the Streamlit app on 127.0.0.1 (localhost only), drives the installed Chrome via
Playwright, writes PNGs to dashboard/screenshots/ and shuts the server down again.
Requires:  pip install playwright   (uses the system Chrome; no browser download)
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "dashboard" / "screenshots"
PORT = 8601
TABS = ["Overview", "Voice of customer", "Drivers", "Segments & risk", "Actions", "Data & model health"]


def trim(path: Path, margin: int = 40) -> None:
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, im.getpixel((5, im.height - 5)))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        im.crop((0, 0, im.width, min(im.height, box[3] + margin))).save(path)


def main() -> None:
    from playwright.sync_api import sync_playwright
    sys.path.insert(0, str(ROOT / "dashboard"))
    import build_static
    SHOTS.mkdir(parents=True, exist_ok=True)
    page = build_static.build()
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"), "--server.port", str(PORT), "--server.headless", "true",
                            "--server.address", "127.0.0.1"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/_stcore/health", timeout=1).status == 200:
                    break
            except Exception:
                time.sleep(1)
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome", headless=True)
            pg = b.new_page(viewport={"width": 1400, "height": 3000})
            pg.goto(f"http://127.0.0.1:{PORT}", wait_until="networkidle", timeout=90000)
            pg.wait_for_selector(".cvi-tiles", timeout=90000)
            pg.wait_for_timeout(2500)
            for i, t in enumerate(TABS, 1):
                pg.get_by_role("tab", name=t).click()
                pg.wait_for_timeout(2500)
                exc = pg.locator('[data-testid="stException"]').count()
                out = SHOTS / f"streamlit_{i}_{t.lower().replace(' & ', '_').replace(' ', '_')}.png"
                pg.screenshot(path=out)
                trim(out)
                print(f"{out.name}: exceptions on page = {exc}")
            # filter demo: Region = North  (KPIs and charts must change)
            pg.get_by_role("tab", name="Overview").click()
            pg.get_by_label("Region").click()
            pg.get_by_role("option", name="North").click()
            pg.wait_for_timeout(2500)
            out = SHOTS / "streamlit_filtered_region_north.png"
            pg.screenshot(path=out)
            trim(out)
            print(f"{out.name}: records in view =", pg.locator("[data-testid='stMetricValue']").first.inner_text())
            pg.close()
            for theme in ["light", "dark"]:
                pg = b.new_page(viewport={"width": 1320, "height": 1000})
                pg.goto(page.as_uri() + f"?theme={theme}", wait_until="load")
                pg.wait_for_function("document.querySelectorAll('.js-plotly-plot').length >= 20", timeout=60000)
                pg.wait_for_timeout(1500)
                out = SHOTS / f"static_dashboard_{theme}.png"
                pg.screenshot(path=out, full_page=True)
                print(out.name)
                pg.close()
            b.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    main()
