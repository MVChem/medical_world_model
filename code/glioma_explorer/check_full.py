"""Verify all patients in the complete HTML and its opaque sandbox preview."""
import asyncio
import base64
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

from .data import DATA_ROOT, catalog, image_info, slice_png

OUT = DATA_ROOT / 'glioma_explorer/runs'


class PreviewHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        if self.path != '/__frame':
            self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' http: https:; style-src 'unsafe-inline'; img-src data: blob: http: https:; connect-src http: https:; sandbox allow-scripts")
        super().end_headers()

    def do_GET(self):
        if self.path == '/__frame':
            body = b'<html><body style="margin:0"><iframe title="Preview" sandbox="allow-scripts" src="/glioma_atlas.html" style="width:100%;height:1100px;border:0"></iframe></body></html>'
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()


async def ready(page):
    await page.wait_for_function('state.patient && document.querySelector("#scan-1").naturalWidth>0 && document.querySelector("#image-loading").hidden', timeout=60000)


async def main():
    report = {'patients_checked': [], 'checks': [], 'errors': []}
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(PreviewHandler, directory=str(OUT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}'
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
        page = await browser.new_page(viewport={'width': 1512, 'height': 1120})
        page.on('pageerror', lambda error: report['errors'].append(str(error)))
        await page.context.set_offline(True)
        await page.goto((OUT / 'glioma_atlas.html').as_uri())
        await ready(page)
        # Compare complete browser volume slicing with the independent Python path.
        await page.evaluate('async()=>{await setPatient("100001");state.overlay=false;}')
        for plane in ['axial', 'coronal', 'sagittal']:
            index = image_info('100001')['default_slices'][plane]
            url = await page.evaluate('async args=>AtlasOffline.render(state.patient,"t1ce",args.plane,args.index,1,false,0)', {'plane': plane, 'index': index})
            actual = np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(',')[1]))).convert('RGB'))
            expected = np.asarray(Image.open(io.BytesIO(slice_png('100001', 't1ce', plane, index, 1, False, 0))).convert('RGB'))
            difference = np.abs(actual.astype(float) - expected.astype(float))
            # Canvas/Pillow nearest-neighbour ties may differ at a resize boundary.
            if actual.shape != expected.shape or difference.mean() > 1:
                raise AssertionError(f'Browser/Python {plane} geometry mismatch: {difference.mean()}')
            report['checks'].append({'plane': plane, 'mean_pixel_difference': float(difference.mean())})
        await page.locator('[data-dataset="mu"]').click()
        await ready(page)
        await page.evaluate('async()=>{await setPatient("PatientID_0191",1);}')
        await ready(page)
        assert '缺少分割' in await page.locator('#mask-note').inner_text()
        assert await page.locator('#followup-panel').is_hidden()
        await page.evaluate('async()=>{await setPatient("PatientID_0003",2,5);}')
        await ready(page)
        assert await page.locator('#scan-time-2').inner_text() == '第 5 次检查'
        await page.locator('[data-plane="sagittal"]').click()
        await ready(page)
        assert int(await page.locator('#slice-slider').get_attribute('max')) == 239
        await page.locator('#slice-slider').fill('210')
        await page.locator('#slice-slider').dispatch_event('input')
        await page.wait_for_timeout(200)
        await ready(page)
        assert '211 / 240' in await page.locator('#slice-value').inner_text()
        await page.locator('#clinical-record').evaluate('(el)=>el.open=true')
        assert await page.locator('#clinical-fields tr').count() > 60
        await page.screenshot(path=str(OUT / 'full-mu.png'), full_page=True)
        await page.locator('[data-dataset="ucsf"]').click()
        await ready(page)
        await page.locator('#file-record').evaluate('(el)=>el.open=true')
        await page.wait_for_function('document.querySelector("#file-image").naturalWidth>0 && document.querySelector("#file-loading").hidden', timeout=60000)
        assert await page.locator('#file-select option').count() == 16
        assert 'subtraction' in await page.locator('#file-select option:checked').inner_text()
        await page.locator('#file-plane').select_option('coronal')
        await page.wait_for_function('document.querySelector("#file-loading").hidden', timeout=60000)
        await page.locator('[data-view="tables"]').click()
        await page.wait_for_function('document.querySelector("#worksheet-body").children.length>0')
        assert await page.locator('#worksheet-select option').count() == 5
        await page.locator('#table-search').fill('100001')
        assert await page.locator('#worksheet-body tr').count() == 2
        await page.locator('[data-dataset="mu"]').click()
        await page.wait_for_function('document.querySelector("#worksheet-select").options.length===7')
        await page.screenshot(path=str(OUT / 'full-tables.png'), full_page=True)
        report['checks'].append('File:// with networking disabled: full planes/slices, nonconsecutive visits, missing masks, all clinical fields, subtraction volumes, original worksheets')
        await page.close()

        # Real chat preview has an opaque sandbox origin. Exercise the same CSP.
        isolated = await browser.new_page(viewport={'width': 1512, 'height': 1120})
        isolated.on('pageerror', lambda error: report['errors'].append(str(error)))
        await isolated.goto(origin + '/__frame')
        frame = isolated.frames[1]
        await ready(frame)
        await frame.evaluate('async()=>{await setPatient("100075");}')
        await ready(frame)
        await frame.locator('[data-sequence="flair"]').click()
        await ready(frame)
        report['checks'].append('Opaque sandbox preview loads local JS volume assets and unit-omission case 100075')
        await isolated.close()

        tasks = [(d, p['id']) for d, cohort in catalog().items() for p in cohort['patients']]
        queue = asyncio.Queue()
        for task in tasks:
            queue.put_nowait(task)

        async def worker():
            context = await browser.new_context(viewport={'width': 1100, 'height': 850})
            await context.set_offline(True)
            page = await context.new_page()
            page.on('pageerror', lambda error: report['errors'].append(str(error)))
            await page.goto((OUT / 'glioma_atlas.html').as_uri())
            await ready(page)
            while not queue.empty():
                try:
                    dataset, pid = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    result = await page.evaluate('''async ([dataset,pid])=>{
                        state.dataset=dataset;
                        await setPatient(pid);
                        await new Promise(resolve=>requestAnimationFrame(resolve));
                        if(state.patient?.id!==pid || !document.querySelector('#image-loading').hidden || !document.querySelector('#scan-1').naturalWidth)
                            throw new Error(pid+': '+document.querySelector('#image-loading').textContent);
                        return {dataset,id:pid,timepoints:state.patient.image.timepoints,files:state.patient.files.length,cache:AtlasOffline.cacheSize()};
                    }''', [dataset, pid])
                    assert result['cache'] <= 8
                    report['patients_checked'].append(result)
                except Exception as error:
                    report['errors'].append(f'{dataset}/{pid}: {error}')
                if len(report['patients_checked']) % 25 == 0:
                    print(f"Browser checked {len(report['patients_checked'])}/501 patients", flush=True)
                    (OUT / 'full_browser_progress.json').write_text(json.dumps({'checked': len(report['patients_checked']), 'errors': report['errors']}))
            await context.close()

        await asyncio.gather(*(worker() for _ in range(4)))
        report['complete'] = False
        (OUT / 'full_browser_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        mobile = await browser.new_page(viewport={'width': 390, 'height': 844})
        await mobile.goto((OUT / 'glioma_atlas.html').as_uri())
        await ready(mobile)
        await mobile.locator('#file-record').evaluate('(el)=>el.open=true')
        await mobile.wait_for_function('document.querySelector("#file-loading").hidden', timeout=60000)
        assert not await mobile.evaluate('document.documentElement.scrollWidth>innerWidth')
        await mobile.screenshot(path=str(OUT / 'full-mobile.png'), full_page=True)
        report['checks'].append('390px mobile layout, including complete file viewer')
        await browser.close()
    server.shutdown()
    report['complete'] = len(report['patients_checked']) == 501 and not report['errors']
    (OUT / 'full_browser_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'complete': report['complete'], 'patients_checked': len(report['patients_checked']), 'checks': report['checks'], 'errors': report['errors']}), flush=True)
    if not report['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
