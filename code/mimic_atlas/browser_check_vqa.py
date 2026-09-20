"""Check VQA browsing against real local data; screenshots stay in runs/."""
import asyncio
import json
import shutil
from pathlib import Path
from playwright.async_api import async_playwright


async def run():
    output = Path(__file__).parent / "runs/vqa_browser_20260918"
    output.mkdir(parents=True, exist_ok=True)
    url = "http://127.0.0.1:8767"
    errors = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=shutil.which("google-chrome") or shutil.which("chromium"), args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(url + "/#vqa")
        await page.wait_for_selector(".vqa-table tbody tr", timeout=120000)
        summary = await (await page.request.get(url + "/api/vqa/summary")).json()
        assert summary["questions"] == 377391
        assert summary["overlap"]["train/valid"] == 698
        await page.get_by_label("官方划分", exact=True).select_option("test")
        await page.wait_for_function("document.querySelector('.vqa-results-heading').textContent.includes('13,793')")
        await page.get_by_label("答案", exact=True).select_option("empty")
        await page.wait_for_function("document.querySelector('.vqa-results-heading').textContent.includes('2,484')")
        assert "空答案" in await page.locator(".vqa-table tbody tr").first.inner_text()
        await page.get_by_label("答案", exact=True).select_option("all")
        await page.wait_for_function("document.querySelector('.vqa-results-heading').textContent.includes('13,793')")
        await page.get_by_label("问答下一页", exact=True).click()
        await page.wait_for_selector('button[aria-label="查看 test:25"]')
        await page.get_by_label("问答上一页", exact=True).click()
        await page.wait_for_selector('button[aria-label="查看 test:0"]')
        await page.screenshot(path=str(output / "vqa_catalog.png"), full_page=True)
        await page.get_by_label("查看 test:0", exact=True).click()
        await page.wait_for_function("document.querySelector('.vqa-image img')?.naturalWidth > 0")
        first = await (await page.request.get(url + "/api/vqa/questions/test/0")).json()
        assert first["image_id"] in await page.locator(".vqa-image img").get_attribute("src")
        await page.wait_for_selector(".vqa-related article")
        siblings = await (await page.request.get(url + "/api/vqa/questions?image_id=" + first["image_id"])).json()
        assert all(row["image_id"] == first["image_id"] for row in siblings["rows"])
        await page.get_by_label("放大胸片", exact=True).click()
        await page.wait_for_function("document.querySelector('.vqa-lightbox img')?.naturalWidth > 0")
        await page.keyboard.press("Escape")
        assert await page.locator(".vqa-lightbox").count() == 0
        await page.locator(".vqa-detail").screenshot(path=str(output / "vqa_detail.png"))
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.screenshot(path=str(output / "vqa_mobile.png"), full_page=True)
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "mobile overflow"
        await page.set_viewport_size({"width": 1440, "height": 1000})
        await page.get_by_role("button", name="查看患者时间线与临床记录 ↗").click()
        await page.wait_for_function("location.hash.includes('subject=')")
        assert "subject=" + first["subject_id"] in page.url
        await page.go_back()
        await page.wait_for_selector(".vqa-table tbody tr")
        await page.locator("#vqa-search").fill("a-question-that-does-not-exist-xyz")
        await page.wait_for_function("document.querySelector('.vqa-table')?.textContent.includes('没有匹配')")
        await page.get_by_role("button", name="重置", exact=True).click()
        await page.wait_for_selector(".vqa-table tbody tr")
        assert not errors, errors
        await browser.close()
    result = {"summary": summary, "browser_errors": errors, "checks": ["source counts and overlap", "split and empty-answer filters", "pagination", "image association", "same-image questions", "enlargement", "patient navigation and back", "empty search and reset", "390px mobile layout"]}
    (output / "checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
