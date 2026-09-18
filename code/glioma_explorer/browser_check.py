"""Smoke-test the real service and the exported HTML in Chrome; save previews."""
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

from .data import DATA_ROOT

ROOT = Path(__file__).resolve().parent
OUT = DATA_ROOT / "glioma_explorer/runs"


async def ready(page):
    await page.wait_for_function("document.querySelector('#scan-1').naturalWidth > 0 && document.querySelector('#image-loading').hidden", timeout=60000)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"checks": [], "browser_errors": []}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
        page = await browser.new_page(viewport={"width": 1512, "height": 1120}, device_scale_factor=1)
        page.on('pageerror', lambda e: result['browser_errors'].append(str(e)))
        # systemctl restart can return before uvicorn finishes importing dependencies.
        for attempt in range(20):
            try:
                r = await page.request.get('http://127.0.0.1:8766/', timeout=1000)
                if r.ok:
                    break
            except Exception:
                if attempt == 19:
                    raise
            await asyncio.sleep(.5)
        await page.goto('http://127.0.0.1:8766', wait_until='networkidle', timeout=60000)
        await ready(page)
        await page.screenshot(path=str(OUT / 'desktop.png'), full_page=True)
        await page.screenshot(path=str(OUT / 'preview.png'))
        assert not await page.evaluate('document.documentElement.scrollWidth > innerWidth')
        result['checks'].append('Desktop layout and actual MRI loaded')

        initial = await page.locator('#scan-1').get_attribute('src')
        await page.locator('[data-sequence="flair"]').click()
        await page.wait_for_function('(before)=>document.querySelector("#scan-1").src!==before && document.querySelector("#image-loading").hidden', arg=initial, timeout=60000)
        assert 'FLAIR' in await page.locator('#caption-1').inner_text()
        result['checks'].append('MRI sequence updates pixels and caption')

        await page.locator('[data-plane="coronal"]').click()
        await ready(page)
        assert await page.locator('.north').first.inner_text() == 'S'
        assert 'CORONAL' in await page.locator('#caption-1').inner_text()
        result['checks'].append('Coronal geometry and orientation labels')

        value = int(await page.locator('#slice-slider').input_value())
        await page.locator('#slice-next').click()
        await page.wait_for_timeout(200)
        await ready(page)
        assert int(await page.locator('#slice-slider').input_value()) == value + 1
        initial = await page.locator('#scan-1').get_attribute('src')
        await page.locator('.toggle-label').click()
        assert not await page.locator('#overlay-toggle').is_checked()
        await page.wait_for_function('(before)=>document.querySelector("#scan-1").src!==before && document.querySelector("#image-loading").hidden', arg=initial, timeout=60000)
        result['checks'].append('Synchronized slice change and segmentation toggle')

        await page.locator('#patient-select').select_option('100001')
        await ready(page)
        assert await page.locator('#detail-id').inner_text() == '100001'
        assert '21.18' in await page.locator('#volume-chart').inner_text()
        result['checks'].append('Second patient with independently checked mask volumes')

        await page.locator('[data-dataset="mu"]').click()
        await page.wait_for_function('document.querySelector("#mu-id").textContent === "PatientID_0003"')
        await ready(page)
        assert await page.locator('#ucsf-viewer').is_visible()
        assert '第 5 次检查' in await page.locator('#mu-timeline').inner_text()
        assert 'NETC' in await page.locator('#legend').inner_text()
        initial = await page.locator('#scan-2').get_attribute('src')
        await page.locator('#visit-second').select_option('5')
        await page.wait_for_function('(before)=>document.querySelector("#scan-2").src!==before && document.querySelector("#image-loading").hidden', arg=initial, timeout=60000)
        assert await page.locator('#scan-time-2').inner_text() == '第 5 次检查'
        assert '286' in await page.locator('#followup-day').inner_text()
        await page.screenshot(path=str(OUT / 'mu-overview.png'), full_page=True)
        await page.locator('#patient-select').select_option('PatientID_0004')
        await ready(page)
        assert await page.locator('#followup-panel').is_hidden()
        assert await page.locator('#visit-second').is_disabled()
        result['checks'].append('MU real MRI, original visits 1/2/5 selection, NETC labels and single-visit patient')
        await page.locator('[data-view="cohort"]').click()
        assert '654' in await page.locator('#fourth-chart-note').inner_text()
        assert await page.locator('#mu-volume-card').is_visible()
        result['checks'].append('MU clinical and scanner record counts remain separate')

        await page.locator('#patient-search').fill('PatientID_0003')
        assert await page.locator('#patient-rows tr[data-id]').count() == 1
        await page.locator('#patient-search').fill('not-a-patient')
        assert '没有匹配' in await page.locator('#patient-rows').inner_text()
        result['checks'].append('Patient search, including empty results')

        response = await page.request.get('http://127.0.0.1:8766/api/export/mu.csv')
        assert response.ok
        assert len((await response.text()).strip().splitlines()) == 204
        result['checks'].append('CSV exports 203 real MU patients')

        mobile = await browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        mobile.on('pageerror', lambda e: result['browser_errors'].append(str(e)))
        await mobile.goto('http://127.0.0.1:8766', wait_until='networkidle', timeout=60000)
        await ready(mobile)
        assert not await mobile.evaluate('document.documentElement.scrollWidth > innerWidth')
        await mobile.screenshot(path=str(OUT / 'mobile.png'), full_page=True)
        result['checks'].append('390 px mobile layout without horizontal page overflow')

        offline = await browser.new_page(viewport={"width": 1512, "height": 1120})
        offline.on('pageerror', lambda e: result['browser_errors'].append(str(e)))
        await offline.context.set_offline(True)
        await offline.goto((OUT / 'glioma_atlas.html').as_uri(), wait_until='load')
        await ready(offline)
        await offline.locator('[data-sequence="t2"]').click()
        await ready(offline)
        assert 'T2' in await offline.locator('#caption-1').inner_text()
        await offline.locator('#slice-next').click()
        await offline.wait_for_timeout(250)
        await ready(offline)
        assert await offline.locator('[data-plane="coronal"]').is_enabled()
        await offline.locator('[data-dataset="mu"]').click()
        await offline.wait_for_function('document.querySelector("#mu-id").textContent === "PatientID_0003"')
        await ready(offline)
        assert await offline.locator('#ucsf-viewer').is_visible()
        assert await offline.locator('#mu-viewer').is_visible()
        assert await offline.locator('#visit-first').is_enabled()
        await offline.locator('[data-sequence="flair"]').click()
        await ready(offline)
        assert 'FLAIR' in await offline.locator('#caption-1').inner_text()
        await offline.locator('#slice-next').click()
        await offline.wait_for_timeout(250)
        await ready(offline)
        result['checks'].append('Complete file:// HTML with local volume assets works with networking disabled')
        await browser.close()
    assert not result['browser_errors'], result['browser_errors']
    (OUT / 'browser_checks.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())
