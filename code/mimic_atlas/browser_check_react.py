"""Exercise React request cancellation and controlled inputs with real data."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

from playwright.async_api import async_playwright


async def run(url, output):
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path=shutil.which("google-chrome") or shutil.which("chromium"),
            args=["--no-sandbox"],
        )
        page = await browser.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(url + "/#subject=12137189")
        await page.locator("#tab-observations").click()
        await page.wait_for_selector(".observation-svg circle")
        search = page.locator("#observation-search")
        await search.press_sequentially("Glucose", delay=100)
        await page.wait_for_function(
            "document.querySelector('#observation-records')?.textContent.includes('Glucose')"
        )
        assert await search.input_value() == "Glucose"
        assert await search.evaluate("el => el === document.activeElement")
        await search.fill("")
        await page.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.includes('165')"
        )
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def delayed(route):
            response = await route.fetch()
            started.set()
            await release.wait()
            try:
                await route.fulfill(response=response)
            finally:
                finished.set()

        await page.route("**/api/patients/12137189/tables/labevents?**", delayed)
        await page.locator("#observation-scope").select_option("patient")
        await asyncio.wait_for(started.wait(), 30)
        await page.evaluate("location.hash = 'subject=10004606'")
        await page.wait_for_function(
            "document.querySelector('#subject-id')?.textContent === '10004606'"
        )
        await page.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.includes('1,665')"
        )
        release.set()
        await asyncio.wait_for(finished.wait(), 30)
        await page.wait_for_timeout(300)
        assert await page.locator("#subject-id").inner_text() == "10004606"
        assert "1,665" in await page.locator("#observation-page").inner_text()
        assert await page.locator(".scan").count() == 0
        await page.locator(".raw-record").first.click()
        raw = json.loads(await page.locator("#raw-row-content").inner_text())
        assert raw["subject_id"] == "10004606"
        await page.keyboard.press("Escape")
        await page.screenshot(path=str(output / "patient_switch.png"), full_page=True)
        await browser.close()
    result = {
        "checks": [
            "Debounced search preserves focus and complete typed text",
            "A delayed response for the previous patient cannot replace the current patient's records",
        ],
        "browser_errors": errors,
    }
    (output / "checks.json").write_text(json.dumps(result, indent=2))
    assert not errors, errors
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8767")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "runs/react_races_20260918",
    )
    args = parser.parse_args()
    asyncio.run(run(args.url, args.output.resolve()))
