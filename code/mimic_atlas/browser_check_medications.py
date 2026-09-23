"""Check medication-pair review against the real selection and original sources.

Run from code/: .venv/bin/python -m mimic_atlas.browser_check_medications
Screenshots and a compact check report stay in a dated runs/ directory.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import shutil
from urllib.parse import quote

from playwright.async_api import async_playwright

from data_preprocessing.medication_events import event_overlaps

REVIEW_PAIR = "medpair:0ba8af3e288dd2243f002a97"
FIRST_PAIR = "medpair:aa99e467f24a63c103dee764"


async def check(url, output):
    output.mkdir(parents=True, exist_ok=True)
    errors, checks = [], []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path=shutil.which("google-chrome") or shutil.which("chromium"),
            args=["--no-sandbox"],
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})
        page.set_default_timeout(30000)
        page.on("pageerror", lambda error: errors.append(str(error)))

        async def read(path):
            response = await page.request.get(url + path, timeout=60000)
            assert response.ok, (path, response.status, await response.text())
            return await response.json()

        async def directory_ready():
            await page.wait_for_selector("#medication-pairs-table tbody tr", timeout=120000)

        async def detail_ready():
            await page.wait_for_function("""() => ['source','target'].every(side => {
                const image = document.querySelector(`[data-endpoint='${side}'] img`);
                return image && image.naturalWidth > 0;
            })""", timeout=120000)
            await page.wait_for_selector("#medication-records-table tbody tr")

        await page.goto(url + "/#medications")
        await directory_ready()
        summary = await read("/api/medication-cohort")
        assert summary["state"] == "ready"
        assert (summary["pairs"], summary["patients"]) == (1022127, 21097)
        assert summary["splits"]["train"]["retained"] == 985153
        assert summary["splits"]["validate"]["retained"] == 10630
        assert summary["splits"]["test"]["retained"] == 26344
        await page.wait_for_function("document.querySelector('.medication-splits').textContent.includes('985,153')")
        assert await page.locator("#medication-pairs-table tbody tr").count() == 25
        await page.screenshot(path=str(output / "directory.png"), full_page=True)
        checks.append("Full selection counts and split labels")

        await page.locator("#medication-split").select_option("test")
        test_rows = await read("/api/medication-cohort/pairs?split=test&limit=25")
        first_test = test_rows["rows"][0]["id"]
        await page.wait_for_selector(f'[data-medication-pair="{first_test}"]')
        assert test_rows["total"] == 26344
        await page.get_by_label("Medication pairs next page", exact=True).click()
        second_page = await read("/api/medication-cohort/pairs?split=test&page=2&limit=25")
        await page.wait_for_selector(f'[data-medication-pair="{second_page["rows"][0]["id"]}"]')
        assert "page=2" in page.url
        await page.get_by_label("Medication pairs previous page", exact=True).click()
        await page.wait_for_selector(f'[data-medication-pair="{first_test}"]')
        await page.locator("#medication-split").select_option("all")
        await page.locator("#medication-subject").fill("10004235")
        await page.get_by_role("button", name="Search pairs", exact=True).click()
        await page.wait_for_function("""() => {
            const rows = [...document.querySelectorAll('#medication-pairs-table tbody tr')];
            return rows.length > 0 && rows.every(row => row.cells[0].textContent.startsWith('10004235'));
        }""")
        checks.append("Split filter, directory pagination, exact patient search")

        await page.goto(url + "/#medications&pair=" + quote(REVIEW_PAIR))
        await detail_ready()
        pair = await read("/api/medication-cohort/pairs/" + quote(REVIEW_PAIR))
        assert pair["patient"] == "10004235"
        assert pair["source_view"] != pair["target_view"] and not pair["full_timeline_adjacent"]
        for side in ("source", "target"):
            endpoint = pair["endpoints"][side]
            assert endpoint["dicom_id"] in await page.locator(f"[data-endpoint='{side}'] img").get_attribute("src")
            assert await page.locator(f"[data-report='{side}']").text_content() == endpoint["report"]
        records = []
        record_page = 1
        while True:
            payload = await read(f"/api/medication-cohort/pairs/{quote(REVIEW_PAIR)}/medications?page={record_page}&limit=100")
            assert payload["counts_match"]
            records.extend(payload["rows"])
            if len(records) >= payload["total"]:
                break
            record_page += 1
        assert len(records) == pair["medication"]["records"] == 369
        assert len({(row["source_table"], row["source_reference"]["patient_row_ordinal"]) for row in records}) == 369
        for row in records:
            assert row["raw_record"]["subject_id"] == pair["patient"]
            assert event_overlaps(row, pair["patient"], pair["source_time"], pair["target_time"])
            assert all(detail["subject_id"] == pair["patient"] for detail in row["raw_details"])
        assert sum(row["source_table"] == "hosp.emar" for row in records) == 98
        assert sum(row["source_table"] == "icu.inputevents" for row in records) == 271
        assert any(row["relative_start_hours"] < 0 for row in records)
        checks.append("Direct nonadjacent AP-to-PA pair: correct image/report identity and all 369 original medication references")
        checks.append("Exact inclusive interval with active infusions starting before source; no admission restriction or padding")

        await page.locator(".medication-raw summary").first.click()
        assert await page.locator(".medication-raw[open] pre").first.is_visible()
        await page.locator(".medication-raw[open] summary").first.click()
        previous_text = await page.locator("#medication-records-table tbody").inner_text()
        await page.get_by_label("Medication records next page", exact=True).click()
        await page.wait_for_function("old => document.querySelector('#medication-records-table tbody')?.textContent.length > 0 && document.querySelector('#medication-records-table tbody').innerText !== old", arg=previous_text)
        await page.get_by_label("Medication records previous page", exact=True).click()
        await page.locator("#medication-source").select_option("hosp.emar")
        await page.wait_for_function("""() => {
            const rows = [...document.querySelectorAll('#medication-records-table tbody tr')];
            return rows.length > 0 && rows.every(row => row.cells[0].textContent.includes('hosp.emar'));
        }""")
        assert "98 results" in await page.locator(".medication-records .medication-pagination").inner_text()
        await page.locator("#medication-record-search").fill("a-medication-name-that-does-not-exist-xyz")
        await page.get_by_role("button", name="Search records", exact=True).click()
        await page.get_by_text("No records match these filters.", exact=True).wait_for()
        await page.locator(".medication-event-filters").get_by_role("button", name="Reset", exact=True).click()
        await page.wait_for_selector("#medication-records-table tbody tr")
        checks.append("Medication pagination, source/name filters, empty state, reset, expanded raw fields")

        await page.get_by_label("Enlarge source chest X-ray", exact=True).click()
        await page.wait_for_function("document.querySelector('#medication-image-dialog[open] img')?.naturalWidth > 0")
        await page.get_by_label("Enlarged image zoom", exact=True).press("End")
        assert await page.get_by_label("Enlarged image zoom", exact=True).input_value() == "3"
        await page.keyboard.press("Escape")
        assert not await page.locator("#medication-image-dialog").is_visible()
        await page.locator(".medication-records-scroll").evaluate("element => { element.scrollLeft = 0; }")
        await page.locator("#medication-pair-detail").screenshot(path=str(output / "pair_detail.png"))
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.locator("#medication-pair-detail").scroll_into_view_if_needed()
        await page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Mobile document overflow"
        await page.set_viewport_size({"width": 1440, "height": 1000})
        checks.append("1800 px image enlargement, zoom, Escape, and 390 px layout")

        await page.get_by_role("button", name="Full patient timeline ↗", exact=True).click()
        await page.wait_for_function("location.hash.includes('subject=10004235')")
        await page.go_back()
        await detail_ready()
        await page.reload()
        await detail_ready()
        assert REVIEW_PAIR in await page.locator(".medication-pair-id").inner_text()
        await page.goto(url + "/#medications&pair=" + quote(FIRST_PAIR))
        await detail_ready()
        assert "Patient 10000032" in await page.locator("#medication-pair-detail h2").inner_text()
        assert "10 source records" in await page.locator(".medication-records .subtle-badge").inner_text()
        await page.goto(url + "/#medications&pair=medpair:does-not-exist")
        await page.get_by_text("Unknown medication pair", exact=True).wait_for()
        assert await page.locator("[data-report]").count() == 0
        checks.append("Patient navigation and Back, reloadable pair links, patient switch, unknown pair without stale reports")
        assert not errors, errors
        await browser.close()
    result = {"state": "passed", "observed_at": datetime.now().astimezone().isoformat(),
              "url": url, "pairs": summary["pairs"], "patients": summary["patients"],
              "review_pair": REVIEW_PAIR, "review_records": len(records),
              "checks": checks, "browser_errors": errors}
    (output / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8767")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "runs" / ("medication_browser_" + datetime.now().strftime("%Y%m%d")))
    args = parser.parse_args()
    asyncio.run(check(args.url.rstrip("/"), args.output))


if __name__ == "__main__":
    main()
