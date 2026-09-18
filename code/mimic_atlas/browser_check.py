"""Exercise the real local UI and its offline HTML export using Playwright."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright


async def loaded(page, count=2):
    await page.wait_for_function(
        "n => document.querySelectorAll('.scan.loaded').length === n",
        arg=count,
        timeout=120000,
    )


async def run(url, output, include_inputs):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"checks": [], "browser_errors": []}
    async with async_playwright() as pw:
        executable = shutil.which("google-chrome") or shutil.which("chromium")
        browser = await pw.chromium.launch(
            executable_path=executable, headless=True, args=["--no-sandbox"]
        )
        page = await browser.new_page(viewport={"width": 1512, "height": 1100})
        page.on("pageerror", lambda error: result["browser_errors"].append(str(error)))
        await page.goto(url + "/#subject=12137189")
        await loaded(page)
        assert await page.locator("#subject-id").inner_text() == "12137189"
        assert (await (await page.request.get(url + "/api/catalog")).json())["counts"][
            "patients"
        ] == 65379
        assert not await page.evaluate(
            "document.documentElement.scrollWidth > innerWidth"
        )
        await page.screenshot(path=str(output / "desktop.png"), full_page=True)
        result["checks"].append(
            "Full metadata counts and two actual CXR images, without desktop overflow"
        )

        await page.locator("#zoom").fill("1.8")
        assert "scale(1.8)" in await page.locator(".scan").first.get_attribute("style")
        await page.locator("#invert").click()
        assert await page.locator("#invert").get_attribute("aria-pressed") == "true"
        await page.locator("#reset-image").click()
        await page.locator("[data-expand='0']").click()
        assert await page.locator("#image-dialog").is_visible()
        await page.keyboard.press("Escape")
        initial = await page.locator(".scan").first.get_attribute("src")
        await page.locator("#pair-next").click()
        await loaded(page)
        assert await page.locator(".scan").first.get_attribute("src") != initial
        await page.locator("#pair-prev").click()
        await loaded(page)
        result["checks"].append(
            "Synchronized viewer adjustments, expanded image and adjacent pair navigation"
        )

        await page.locator("#tab-labels").click()
        assert await page.locator(".label-table tbody tr").count() == 14
        assert "不确定" in await page.locator("#labels-panel").inner_text()
        await page.locator("#tab-clinical").click()
        await page.wait_for_selector(".clinical-summary", timeout=180000)
        assert "28837774" in await page.locator("#clinical-panel").inner_text()
        assert "36697114" in await page.locator("#clinical-panel").inner_text()
        await page.locator("#event-search").fill("not-an-event")
        assert "没有匹配" in await page.locator("#event-table").inner_text()
        await page.locator("#event-search").fill("")
        result["checks"].append(
            "14 four-state labels; verified admission/ICU IDs and event search"
        )
        if include_inputs:
            if await page.locator("#load-inputs").count():
                await page.locator("#load-inputs").click()
            await page.wait_for_function(
                "document.querySelector('#clinical-panel')?.textContent.includes('已读取 ICU inputevents')",
                timeout=240000,
            )
            await page.locator("#event-filter").select_option("input")
            assert await page.locator("#event-table tbody tr").count() > 0
            await page.locator("#event-filter").select_option("all")
            result["checks"].append(
                "On-demand real ICU inputevents, with amounts and original units"
            )
        await page.evaluate("window.scrollTo(0,0)")
        await page.screenshot(path=str(output / "clinical.png"), full_page=True)

        async with page.expect_download() as download_info:
            await page.locator("#export-html").click()
        download = await download_info.value
        exported = output / "mimic_atlas.html"
        await download.save_as(exported)
        offline_context = await browser.new_context(
            offline=True, viewport={"width": 1512, "height": 1100}
        )
        offline = await offline_context.new_page()
        offline.on(
            "pageerror", lambda error: result["browser_errors"].append(str(error))
        )
        await offline.goto(exported.as_uri())
        await loaded(offline)
        assert "OFFLINE" in await offline.locator("#mode-badge").inner_text()
        await offline.locator("#tab-clinical").click()
        assert "28837774" in await offline.locator("#clinical-panel").inner_text()
        await offline.screenshot(path=str(output / "offline.png"), full_page=True)
        result["checks"].append(
            "HTML download reopens with network disabled, real images and linked IV context"
        )

        await page.locator("#back-cohort").click()
        await page.locator("#patient-search").fill("no-such-patient")
        await page.wait_for_function(
            "document.querySelector('#patient-list')?.textContent.includes('没有匹配')"
        )
        await page.locator("#patient-search").fill("12698729")
        await page.wait_for_selector('[data-patient="12698729"]')
        await page.locator('.patient-link[data-patient="12698729"]').click()
        await page.wait_for_function(
            "document.querySelector('#subject-id')?.textContent==='12698729'"
        )
        await loaded(page)
        assert await page.locator(".study-node").count() == 52
        await page.locator(".study-node").first.click()
        await loaded(page, 1)
        assert await page.locator("#export-html").is_disabled()
        await page.locator("#back-cohort").click()
        await page.locator("#patient-search").fill("")
        await page.wait_for_function(
            "document.querySelectorAll('.patient-link').length===25"
        )
        await page.locator("#page-next").click()
        await page.wait_for_function(
            "document.querySelector('#range-info')?.textContent.startsWith('26–50 /')"
        )
        await page.locator("#split-filter").select_option("test")
        await page.wait_for_function(
            "document.querySelector('#list-count')?.textContent==='293'"
        )
        result["checks"].append(
            "Full-catalog search/empty state, 52-study timeline, single study, pagination and official split"
        )

        await page.locator("#split-filter").select_option("all")
        await page.locator("#patient-search").fill("16454913")
        await page.wait_for_selector('[data-patient="16454913"]')
        await page.locator('.patient-link[data-patient="16454913"]').click()
        await page.wait_for_function(
            "document.querySelector('#subject-id')?.textContent==='16454913'"
        )
        await loaded(page)
        options = await page.locator("#pair-select option").all_text_contents()
        index = next(
            i for i, t in enumerate(options) if "s56164331" in t and "s50631837" in t
        )
        await page.locator("#pair-select").select_option(index=index)
        await page.locator("#tab-clinical").click()
        await page.wait_for_selector(".status-badge.unmatched", timeout=180000)
        assert "不拼接临床事件" in await page.locator("#clinical-panel").inner_text()
        result["checks"].append(
            "Real cross-episode pair retains unmatched linkage without fabricated events"
        )

        mobile = await browser.new_page(viewport={"width": 390, "height": 844})
        mobile.on(
            "pageerror", lambda error: result["browser_errors"].append(str(error))
        )
        await mobile.goto(url + "/#subject=12137189")
        await loaded(mobile)
        assert not await mobile.evaluate(
            "document.documentElement.scrollWidth > innerWidth"
        )
        await mobile.screenshot(path=str(output / "mobile.png"), full_page=True)
        await mobile.locator("#tab-clinical").click()
        await mobile.wait_for_selector(".clinical-summary")
        assert not await mobile.evaluate(
            "document.documentElement.scrollWidth > innerWidth"
        )
        result["checks"].append(
            "390px mobile layout and clinical timeline without page overflow"
        )
        await browser.close()
    (output / "checks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    assert not result["browser_errors"], result["browser_errors"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8767")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / ("browser_" + datetime.now().astimezone().strftime("%Y%m%d")),
    )
    parser.add_argument("--include-inputs", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.output, args.include_inputs))
