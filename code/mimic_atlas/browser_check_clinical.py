"""Real-source checks for all extended tables, trends, CSV and offline HTML."""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright


async def check_delayed_admissions(browser, url, checks):
    """Clinical tables may finish before the background admission request."""
    page = await browser.new_page()
    page.on("pageerror", lambda error: checks["browser_errors"].append(str(error)))
    pending = True

    async def delay_background(route):
        path = route.request.url
        if pending and any(
            part in path
            for part in ("?compact=true", "/selection?", "/clinical-status")
        ):
            response = await route.fetch()
            body = await response.json()
            status = {"state": "loading", "stage": "Delayed background response"}
            if path.endswith("/clinical-status"):
                body = status
            else:
                body["admissions"] = []
                body["clinical_status"] = status
            await route.fulfill(response=response, json=body)
        else:
            await route.continue_()

    await page.route("**/api/patients/12137189**", delay_background)
    await page.goto(url + "/#subject=12137189")
    await page.locator("#tab-observations").click()
    await page.wait_for_selector("#observation-table")
    assert await page.locator("#observation-admission option").count() == 1
    await page.locator("#observation-table").select_option("chartevents")
    await page.locator("#observation-scope").select_option("patient")
    pending = False
    await page.wait_for_function(
        "document.querySelectorAll('#observation-admission option').length > 1"
    )
    assert await page.locator("#observation-table").input_value() == "chartevents"
    assert await page.locator("#observation-scope").input_value() == "patient"
    await page.locator("#observation-scope").select_option("admission")
    await page.locator("#observation-admission").select_option("28837774")
    checks["checks"].append(
        "Late admission data updates filters without reload or losing the selected table/scope"
    )
    await page.close()


async def run(url, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks = {"checks": [], "browser_errors": []}
    async with async_playwright() as pw:
        executable = shutil.which("google-chrome") or shutil.which("chromium")
        browser = await pw.chromium.launch(
            executable_path=executable, headless=True, args=["--no-sandbox"]
        )
        page = await browser.new_page(viewport={"width": 1512, "height": 1100})
        page.on("pageerror", lambda error: checks["browser_errors"].append(str(error)))
        await page.goto(url + "/#subject=12137189")
        await page.wait_for_function(
            "document.querySelector('#subject-id')?.textContent==='12137189'",
            timeout=120000,
        )
        await page.locator("#tab-observations").click()
        catalog = await (await page.request.get(url + "/api/catalog")).json()
        assert catalog["storage"]["mode"] == "patient_index"
        before = await (
            await page.request.get(url + "/api/patients/12137189/tables")
        ).json()
        assert all(m["indexed"] and isinstance(m["count"], int) for m in before)
        started = time.monotonic()
        assert all(m["state"] != "not_loaded" for m in before)
        assert not await page.locator("#load-one-table").is_visible()
        assert not await page.locator("#load-all-tables").is_visible()
        # A GET opening the patient alone prepares every table, without load clicks/POSTs.
        for _ in range(240):
            manifest = await (
                await page.request.get(url + "/api/patients/12137189/tables")
            ).json()
            assert not any(r["state"] in {"error", "unavailable"} for r in manifest), (
                manifest
            )
            if all(r["state"] == "ready" for r in manifest):
                break
            await asyncio.sleep(2.5)
        else:
            raise AssertionError("Extended scans did not finish within 10 minutes")
        checks["table_counts"] = {r["name"]: r["count"] for r in manifest}
        assert checks["table_counts"]["labevents"] == 608
        assert checks["table_counts"]["chartevents"] == 18306
        assert len(manifest) == 17
        assert all(
            m["rows_scanned"] == 0 and m["read_method"] == "patient_index"
            for m in manifest
        )
        checks["table_load_seconds"] = round(time.monotonic() - started, 3)
        checks["checks"].append(
            "Opening the patient automatically loads all 17 tables; no manual load buttons or POSTs; zero source CSV rows scanned"
        )

        await page.reload()
        await page.wait_for_selector(".scan.loaded")
        await page.locator("#tab-observations").click()
        await page.wait_for_selector(".observation-svg circle")
        await page.locator("#observation-scope").select_option("patient")
        await page.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.includes('608')"
        )
        await page.locator("#observation-next").click()
        await page.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.startsWith('2 /')"
        )
        await page.locator(".raw-record").first.click()
        assert await page.locator("#observation-row-dialog").is_visible()
        assert "labevent_id" in await page.locator("#raw-row-content").inner_text()
        await page.keyboard.press("Escape")
        checks["checks"].append(
            "Complete lab pagination and full original-row inspection"
        )

        await page.locator("#observation-scope").select_option("admission")
        await page.locator("#observation-admission").select_option("28837774")
        await page.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.includes('514')"
        )
        data = await (
            await page.request.get(
                url
                + "/api/patients/12137189/tables/labevents?scope=admission&hadm_id=28837774"
            )
        ).json()
        assert data["total"] == 514 and all(
            r["hadm_id"] == "28837774" for r in data["rows"]
        )
        checks["checks"].append(
            "Exact admission filtering keeps 514 labs, excluding 75 unassigned and 19 other-admission records"
        )

        await page.locator("#observation-scope").select_option("patient")
        await page.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.includes('608')"
        )
        async with page.expect_download() as download_info:
            await page.locator("#observation-csv").click()
        download = await download_info.value
        csv_path = output / "labevents.csv"
        await download.save_as(csv_path)
        rows = list(csv.DictReader(io.StringIO(csv_path.read_text())))
        assert len(rows) == 608 and sum(not r["hadm_id"] for r in rows) == 75
        assert sum(r["hadm_id"] == "28837774" for r in rows) == data["total"]
        checks["checks"].append(
            "CSV exports all 608 unmodified rows, including missing admission identifiers"
        )

        await page.locator("#observation-search").fill("not-a-clinical-record-123")
        await page.wait_for_function(
            "document.querySelector('#observation-records')?.textContent.includes('没有匹配')"
        )
        await page.locator("#observation-search").fill("")
        await page.locator("#observation-table").select_option("chartevents")
        await page.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.includes('18,306')"
        )
        metric = await page.locator("#observation-series option").evaluate_all(
            "opts=>opts.find(o=>o.textContent.startsWith('Heart Rate · bpm'))?.value"
        )
        assert metric
        await page.locator("#observation-series").select_option(metric)
        await page.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.includes('293')"
        )
        assert "Heart Rate bpm" in await page.locator(".observation-svg").text_content()
        assert await page.locator(".observation-svg circle").count() == 293
        # Last SVG point is topmost even when observations overlap.
        await page.locator(".observation-svg circle").last.hover()
        assert (
            "录入:"
            in await page.locator(".observation-svg circle title").last.text_content()
        )
        await page.locator("#observations-panel").scroll_into_view_if_needed()
        await page.screenshot(path=str(output / "vitals.png"))
        checks["checks"].append(
            "293 heart-rate observations in bpm, timestamped points with original values and storetime tooltips"
        )

        await page.locator("#observation-table").select_option("emar_detail")
        await page.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.includes('714')"
        )
        assert (
            "emar.charttime" in await page.locator("#observation-records").inner_text()
        )
        await page.locator("#observation-table").select_option("patients")
        await page.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.endsWith('1 条')"
        )
        assert "无事件时刻" in await page.locator("#observation-records").inner_text()
        await page.locator("#observation-table").select_option("omr")
        await page.wait_for_function(
            "document.querySelector('#observation-records')?.textContent.includes('此患者在该表中没有记录')"
        )
        checks["checks"].append(
            "eMAR child rows receive unique parent times; undated metadata and true empty OMR are distinct"
        )

        await page.locator("#observation-table").select_option("labevents")
        await page.locator("#observation-scope").select_option("window")
        await page.wait_for_selector(".observation-svg circle")
        await page.locator("#observations-panel").scroll_into_view_if_needed()
        await page.screenshot(path=str(output / "labs.png"))
        async with page.expect_download() as download_info:
            await page.locator("#export-html").click()
        exported = output / "mimic_clinical_atlas.html"
        await (await download_info.value).save_as(exported)
        offline_context = await browser.new_context(
            offline=True, viewport={"width": 1512, "height": 1100}
        )
        offline = await offline_context.new_page()
        offline.on(
            "pageerror", lambda error: checks["browser_errors"].append(str(error))
        )
        await offline.goto(exported.as_uri())
        await offline.wait_for_selector(".scan.loaded")
        await offline.locator("#tab-observations").click()
        await offline.wait_for_selector(".observation-svg circle")
        assert await offline.locator("#observation-scope").is_disabled()
        assert "165" in await offline.locator("#observation-page").inner_text()
        await offline.locator("#observation-table").select_option("chartevents")
        await offline.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.includes('3,559')"
        )
        assert await offline.locator(".observation-svg circle").count() > 0
        await offline.locator("#observations-panel").scroll_into_view_if_needed()
        await offline.screenshot(path=str(output / "offline.png"))
        checks["checks"].append(
            "Network-disabled HTML retains 165 labs and 3,559 ICU rows for the exported ±24h window"
        )

        mobile = await browser.new_page(viewport={"width": 390, "height": 844})
        mobile.on(
            "pageerror", lambda error: checks["browser_errors"].append(str(error))
        )
        await mobile.goto(url + "/#subject=12137189")
        await mobile.wait_for_selector(".scan.loaded")
        await mobile.locator("#tab-observations").click()
        await mobile.wait_for_selector(".observation-svg circle")
        assert not await mobile.evaluate(
            "document.documentElement.scrollWidth > innerWidth"
        )
        await mobile.locator("#observations-panel").scroll_into_view_if_needed()
        await mobile.screenshot(path=str(output / "mobile.png"))
        checks["checks"].append(
            "Mobile filters, plot and raw rows fit 390px without page overflow"
        )
        await check_delayed_admissions(browser, url, checks)
        await browser.close()
    (output / "checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2)
    )
    assert not checks["browser_errors"], checks["browser_errors"]
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8767")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / ("clinical_browser_" + datetime.now().astimezone().strftime("%Y%m%d")),
    )
    args = parser.parse_args()
    asyncio.run(run(args.url, args.output))
