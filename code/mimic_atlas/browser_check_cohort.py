"""Real-data all-patient browsing and remote bandwidth checks."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright


async def run(url, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"checks": [], "browser_errors": [], "traffic": {}}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path=shutil.which("google-chrome") or shutil.which("chromium"),
            args=["--no-sandbox"],
        )
        context = await browser.new_context(viewport={"width": 1512, "height": 1050})
        page = await context.new_page()
        page.on("pageerror", lambda e: result["browser_errors"].append(str(e)))
        requests = []
        page.on("request", lambda r: requests.append((r.method, r.url)))
        session = await context.new_cdp_session(page)
        await session.send("Network.enable")
        transferred = []
        session.on(
            "Network.loadingFinished",
            lambda e: transferred.append(e["encodedDataLength"]),
        )
        await page.goto(url)
        await page.wait_for_selector(
            "#overview-content", timeout=180000
        )
        await page.wait_for_function(
            "document.querySelector('#metric-patients')?.textContent==='368,138'",
            timeout=180000,
        )
        await page.wait_for_timeout(300)
        result["traffic"]["home_wire_bytes"] = sum(transferred)
        assert not any(
            "/api/images/" in u or "/api/patients" in u or "/api/cohort/pairs" in u
            for _, u in requests
        ), requests
        assert await page.locator(".patient-link").count() == 0
        assert await page.locator("#cohort-content").count() == 0
        assert await page.locator("h1").inner_text() == "数据总览"
        assert await page.locator("#nav-overview").get_attribute("aria-current") == "page"
        assert await page.locator("#patient-content").is_hidden()
        assert (
            await page.evaluate(
                "getComputedStyle(document.querySelector('.sidebar')).backgroundColor"
            )
            == "rgb(252, 252, 253)"
        )
        assert not await page.evaluate(
            "document.documentElement.scrollWidth>innerWidth"
        )
        await page.screenshot(path=str(output / "overview.png"), full_page=True)
        result["checks"].append(
            "Standalone overview shows 368,138 union patients; no list, patient-detail or image requests"
        )
        assert result["traffic"]["home_wire_bytes"] < 160000
        await page.locator('[data-coverage="matched"]').click()
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='61,868'"
        )
        assert await page.locator("h1").inner_text() == "全量患者"
        assert await page.locator("#coverage-filter").input_value() == "matched"
        assert await page.locator("#dataset-intro, .metrics, .cohort-summary").count() == 0
        await page.reload()
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='61,868'"
        )
        assert await page.locator("#coverage-filter").input_value() == "matched"
        await page.go_back()
        await page.wait_for_selector("#overview-content")
        await page.locator('[data-split="test"]').click()
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='1,991'"
        )
        assert await page.locator("h1").inner_text() == "影像配对"
        assert await page.locator("#split-filter").input_value() == "test"
        assert await page.locator("#dataset-intro, .metrics, .cohort-summary").count() == 0
        await page.locator("#nav-featured").click()
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='7'"
        )
        assert await page.locator("h1").inner_text() == "精选示例"
        assert await page.locator("#dataset-intro, .metrics, .cohort-summary").count() == 0
        await page.screenshot(path=str(output / "featured.png"), full_page=True)
        result["checks"].append(
            "Patients, pairs and featured pages have distinct headings and lists without repeated overview; overview links seed filters and survive reload/history"
        )
        await page.locator("#nav-patients").click()
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='368,138'"
        )
        assert await page.locator(".patient-link").count() == 25
        await page.screenshot(path=str(output / "patient_list.png"), full_page=True)
        first = await page.locator(".patient-link").first.inner_text()
        await page.locator("#page-next").click()
        await page.wait_for_function(
            "document.querySelector('#range-info')?.textContent.startsWith('26–50 /')"
        )
        assert await page.locator(".patient-link").first.inner_text() != first
        await page.locator("#coverage-filter").select_option("matched")
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='61,868'"
        )
        await page.locator("#paired-filter").select_option("linked")
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='14,119'"
        )
        result["checks"].append(
            "Full pagination plus cross-source and same-admission patient filtering"
        )
        await page.locator("#nav-pairs").click()
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='110,729'"
        )
        await page.locator("#split-filter").select_option("test")
        await page.locator("#linkage-filter").select_option("unique")
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='1,159'"
        )
        await page.screenshot(path=str(output / "pair_candidates.png"), full_page=True)
        result["checks"].append(
            "All 110,729 pair candidates; 1,159 uniquely matched official test pairs"
        )
        # Open a non-curated patient directly from the pair table.
        target = page.locator("[data-transition]").first
        selected = await target.get_attribute("data-transition")
        pid = await target.get_attribute("data-patient")
        before = sum(transferred)
        await target.click()
        await page.wait_for_function(
            "document.querySelectorAll('.scan.loaded').length===2", timeout=180000
        )
        assert selected in page.url
        assert await page.locator("#subject-id").inner_text() == pid
        images = await page.locator(".scan").evaluate_all(
            "(els)=>els.map(e=>({src:e.src,width:e.naturalWidth,height:e.naturalHeight}))"
        )
        assert all(
            "size=512" in im["src"] and max(im["width"], im["height"]) <= 512
            for im in images
        )
        assert not any(method == "POST" for method, _ in requests)
        result["traffic"]["selected_pair_wire_bytes"] = sum(transferred) - before
        assert result["traffic"]["selected_pair_wire_bytes"] < 250000
        await page.locator("#tab-labels").click()
        assert await page.locator(".label-table tbody tr").count() == 14
        await page.screenshot(path=str(output / "patient.png"), full_page=True)
        await page.locator('[data-expand="0"]').click()
        await page.wait_for_function(
            "document.querySelector('#dialog-image').naturalWidth>512"
        )
        assert "size=1800" in await page.locator("#dialog-image").get_attribute("src")
        await page.keyboard.press("Escape")
        result["checks"].append(
            "Non-featured pair opens exact selection; 512px WebP previews, reports/labels loaded only for selection; larger image on click"
        )
        await page.locator("#back-cohort").click()
        await page.locator("#nav-patients").click()
        await page.locator("#coverage-filter").select_option("iv_only")
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='302,759'"
        )
        await page.locator("#patient-search").fill("10000068")
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='1'"
        )
        await page.locator(".patient-link").first.click()
        await page.wait_for_selector(
            "#observation-table", state="visible", timeout=120000
        )
        assert await page.locator("#iv-only-note").is_visible()
        assert await page.locator("#observation-scope").input_value() == "patient"
        assert await page.locator(".scan").count() == 0
        await page.locator("#observation-table").select_option("patients")
        assert not await page.locator("#load-one-table").is_visible()
        await page.wait_for_selector(".observation-records tbody tr", timeout=120000)
        assert await page.locator(".observation-records tbody tr").count() == 1
        await page.screenshot(path=str(output / "iv_only.png"), full_page=True)
        result["checks"].append(
            "IV-only patient detail without CXR; patient metadata loads automatically; no phantom images or timestamps"
        )
        await page.locator("#tab-clinical").click()
        await page.wait_for_selector(".raw-context", timeout=120000)
        await page.locator(".raw-context > summary").click()
        await page.wait_for_selector(".raw-context-row", timeout=30000)
        await page.locator("#raw-context-table").select_option("diagnoses_icd")
        await page.wait_for_function(
            "document.querySelector('#raw-context-rows')?.textContent.includes('icd_code')"
        )
        await page.locator(".raw-context-row > summary").first.click()
        assert (
            "icd_version"
            in await page.locator(".raw-context-row[open] pre").inner_text()
        )
        result["checks"].append(
            "IV-only inpatient diagnoses and complete original fields remain browsable without an image pair"
        )
        await page.locator("#back-cohort").click()
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.locator("#nav-patients").click()
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='368,138'"
        )
        assert not await page.evaluate(
            "document.documentElement.scrollWidth>innerWidth"
        )
        await page.screenshot(path=str(output / "mobile.png"), full_page=True)
        await page.locator("#patient-search").fill("12137189")
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='1'"
        )
        await page.locator(".patient-link").click()
        await page.wait_for_function(
            "document.querySelectorAll('.scan.loaded').length===2", timeout=120000
        )
        assert not await page.evaluate(
            "document.documentElement.scrollWidth>innerWidth"
        )
        await page.screenshot(path=str(output / "mobile_patient.png"), full_page=True)
        result["checks"].append(
            "390px mobile: all-patient filters, horizontal table and selected patient without page overflow"
        )
        await browser.close()
    (output / "checks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    assert not result["browser_errors"], result["browser_errors"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8767")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "runs"
        / ("cohort_browser_" + datetime.now().astimezone().strftime("%Y%m%d")),
    )
    args = parser.parse_args()
    asyncio.run(run(args.url, args.output))
