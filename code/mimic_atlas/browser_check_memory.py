"""Real-source cache eviction/reload and migrated export navigation checks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path

from playwright.async_api import async_playwright


async def run(url, output):
    output.mkdir(parents=True, exist_ok=True)
    result = {"checks": [], "browser_errors": []}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path=shutil.which("google-chrome") or shutil.which("chromium"),
            args=["--no-sandbox"],
        )
        page = await browser.new_page(viewport={"width": 1512, "height": 1050})
        page.on("pageerror", lambda e: result["browser_errors"].append(str(e)))

        async def get(path):
            response = await page.request.get(url + path)
            assert response.ok, (path, response.status, await response.text())
            return await response.json()

        async def wait_patient(pid):
            for _ in range(240):
                manifest = await get(f"/api/patients/{pid}/tables")
                status = await get(f"/api/patients/{pid}/clinical-status")
                assert not any(m["state"] == "error" for m in manifest), manifest
                assert status["state"] != "error", status
                if (
                    all(m["state"] == "ready" for m in manifest)
                    and status["state"] == "ready"
                ):
                    return
                await asyncio.sleep(0.5)
            raise AssertionError(f"Patient {pid} did not finish")

        await page.goto(url + "/#subject=12137189")
        await page.locator("#tab-observations").click()
        await page.locator("#observation-scope").select_option("patient")
        await wait_patient("12137189")
        await page.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.includes('608')"
        )
        original = await get(
            "/api/patients/12137189/tables/labevents?scope=patient&limit=200"
        )
        # Wait for the browser's initial base-record polling to finish before eviction.
        await page.wait_for_timeout(1500)
        state = await get("/api/memory")
        count = state["patients"]["max_patients"]
        assert count <= 10, "Run this check with a small patient-cache count"
        others = await get(f"/api/patients?coverage=iv_only&limit={count}")
        snapshots = []
        for row in others["rows"]:
            pid = row["subject_id"]
            await get(f"/api/patients/{pid}?compact=true")
            await wait_patient(pid)
            state = await get("/api/memory")
            assert state["patients"]["patients"] <= count
            snapshots.append(state["patients"])
        assert "12137189" not in state["patients"]["subjects"], state
        # Existing tab must recover without a reload, while preserving filters.
        await page.locator("#observation-next").click()
        await page.wait_for_function(
            "document.querySelector('#observation-page')?.textContent.startsWith('2 /') && document.querySelector('#observation-page')?.textContent.includes('608')",
            timeout=90000,
        )
        await wait_patient("12137189")
        reloaded = await get(
            "/api/patients/12137189/tables/labevents?scope=patient&limit=200"
        )
        assert reloaded["rows"] == original["rows"]
        assert await page.locator("#observation-scope").input_value() == "patient"
        result["patient_cache_samples"] = snapshots
        result["checks"].append(
            "LRU patient count stays bounded; evicted open tab automatically reloads the same original rows and preserves scope/page"
        )

        # A unique encoding variant ensures the first call measures a cache miss.
        image = "/api/images/0ffdea1c-2ea8916c-5e3fd1be-be1a29f8-f10379ec?size=511&format=webp&quality=61"
        before = await get("/api/memory")
        timings, payloads = [], []
        for _ in range(2):
            start = time.monotonic()
            response = await page.request.get(url + image)
            assert response.ok
            payloads.append(await response.body())
            timings.append(round((time.monotonic() - start) * 1000, 2))
        after = await get("/api/memory")
        assert payloads[0] == payloads[1]
        assert after["images"]["hits"] > before["images"]["hits"]
        assert after["images"]["bytes"] <= after["images"]["max_bytes"]
        result["image_request_ms"] = timings
        result["image_sha256"] = hashlib.sha256(payloads[0]).hexdigest()
        result["checks"].append(
            "Repeated preview reuses identical encoded bytes within its memory budget"
        )

        bundles = (await get("/api/exports"))["bundles"]
        assert sorted(b["cases"] for b in bundles) == [4, 4, 10, 10]
        matches = {}
        for bundle in bundles:
            rows = (await get(f"/api/exports/{bundle['id']}/cases"))["rows"]
            for row in rows:
                pid = row["subject_id"]
                if pid not in matches:
                    first = await get(f"/api/cohort/pairs?q={pid}&limit=100")
                    pairs = first["rows"]
                    for n in range(2, (first["total"] + 99) // 100 + 1):
                        pairs += (
                            await get(f"/api/cohort/pairs?q={pid}&limit=100&page={n}")
                        )["rows"]
                    matches[pid] = {p["transition_id"] for p in pairs}
                assert row["transition_id"] in matches[pid], row
        await page.goto(url + "/#exports")
        await page.wait_for_selector(".export-bundle")
        assert await page.locator(".export-bundle").count() == 4
        first = page.locator(".export-bundle").first
        await first.get_by_role("button", name="浏览片段").click()
        target = first.locator("[data-export-transition]").first
        transition = await target.get_attribute("data-export-transition")
        await page.screenshot(path=str(output / "exports.png"), full_page=True)
        await target.click()
        await page.wait_for_function(
            "document.querySelectorAll('.scan.loaded').length===2", timeout=90000
        )
        assert transition in page.url
        assert await page.locator("#subject-id").inner_text() == "12137189"
        await page.locator("#back-cohort").click()
        await page.wait_for_selector(".export-bundle")
        await page.set_viewport_size({"width": 390, "height": 844})
        assert not await page.evaluate(
            "document.documentElement.scrollWidth>innerWidth"
        )
        await page.screenshot(path=str(output / "exports_mobile.png"), full_page=True)
        bundle = bundles[0]
        filename = "transitions.jsonl"
        response = await page.request.get(
            url + f"/api/exports/{bundle['id']}/files/{filename}"
        )
        assert response.ok and "attachment" in response.headers["content-disposition"]
        original_file = (
            Path(__file__).parent
            / "runs/exports"
            / Path(*bundle["id"].split("~"))
            / filename
        )
        assert await response.body() == original_file.read_bytes()
        result["checks"].append(
            "All four migrated bundles preserve exact pair IDs; desktop/mobile navigation and byte-identical original downloads work"
        )
        result["memory_after"] = await get("/api/memory")
        await browser.close()
    (output / "checks.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    assert not result["browser_errors"], result["browser_errors"]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8767")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "runs/memory_browser_20260918",
    )
    args = parser.parse_args()
    asyncio.run(run(args.url, args.output.resolve()))
