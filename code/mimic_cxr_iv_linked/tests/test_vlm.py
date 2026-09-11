import argparse
import asyncio
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import sys
import threading
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vlm_protocol import parse_response,request,review_flags,SCHEMA
from screen_vlm import run
from summarize_vlm import run as summarize
from vlm_triage import triage


def label():
    return dict(primary_class='stable',direction='not_applicable',stable_state='persistent_abnormalities',
        image_assessment='stable',report_assessment='stable',device_change='no',technical_confound='no',
        confidence='medium',image_evidence='Similar bilateral opacities.',report_evidence='Both reports describe unchanged opacities.')


def response(finish='stop'):
    return {'choices':[{'finish_reason':finish,'message':{'content':json.dumps(label())}}],'usage':{'prompt_tokens':2400,'completion_tokens':170}}


def test_incomplete_generation_not_counted_as_class():
    with pytest.raises(ValueError,match='Noncomplete'):
        parse_response(response('length'))
    value=label()
    value['image_assessment']='changed'
    assert 'stable_with_change_evidence' in review_flags(value)
    assert 'image_report_disagreement' in review_flags(value)
    assert triage(value,review_flags(value))['screening_class']=='needs_review'
    assert triage(label(),[])['screening_class']=='stable'
    value['confidence']='high'
    assert triage(value,review_flags(value))['screening_class']=='needs_review'


def test_client_resume_and_denominators(tmp_path):
    seen=[]
    seen_lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({'data':[{'id':'mimic-qwen35-9b'}]}).encode())
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with seen_lock:
                seen.append(body)
                incomplete=len(seen)==1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response('length' if incomplete else 'stop')).encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    rows=[]
    for i in range(2):
        rows.append(dict(pair_id=str(i),subject_id='patient',split='train',source_study='a',target_study='b',
            source_view='AP',target_view='PA',source_path=str(tmp_path/'a.jpg'),target_path=str(tmp_path/'b.jpg'),
            source_image='a',target_image='b',source_report='FINDINGS: '+('word '*500),target_report='IMPRESSION: unchanged',
            realized_gap_hours=2,same_view=False,horizon_bin='0-6h',link_status='same_admission',original_strict_subset=False,tiers={}))
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    a=argparse.Namespace(out=tmp_path/'results',manifest=manifest,endpoints=[f'http://127.0.0.1:{server.server_port}'],
        scope='expanded',train_only=False,limit=0,model='mimic-qwen35-9b',model_snapshot='test',concurrency=2,retries=2)
    try:
        asyncio.run(run(a))
        old=json.loads((a.out/'config.json').read_text())
        old.pop('incomplete_output_retry_cap')
        old.pop('incomplete_output_retry_evidence_max_chars')
        (a.out/'config.json').write_text(json.dumps(old))
        asyncio.run(run(a))
    finally:
        server.shutdown()
        server.server_close()
    assert len(seen)==3 # one incomplete attempt repaired, resume repeats nothing
    assert (a.out/'config_before_length_retry.json').exists()
    repaired=[r for r in seen if r['max_tokens']>512]
    assert len(repaired)==1
    assert repaired[0]['response_format']['json_schema']['schema']['properties']['image_evidence']['maxLength']==240
    assert 'maxLength' not in SCHEMA['properties']['image_evidence'] # no shared schema mutation
    for body in seen:
        content=body['messages'][1]['content']
        assert sum(r['type']=='image_url' for r in content)==2
        assert rows[0]['source_report'] in '\n'.join(r.get('text','') for r in content)
        assert body['chat_template_kwargs']['enable_thinking'] is False
    summarize(manifest,a.out,tmp_path/'summary')
    stats=json.loads((tmp_path/'summary/statistics.json').read_text())
    assert stats['complete'] is True
    assert stats['groups']['all']['completed_unique_patients']==1
    assert stats['groups']['all']['primary_classes']['stable']['n']==2
    assert 'original_70318_subset' not in stats['groups']
    # An unprocessed row is a coverage gap, not an indeterminate or stable label.
    with sqlite3.connect(a.out/'results.sqlite') as db:
        db.execute("DELETE FROM results WHERE pair_id='1'")
    summarize(manifest,a.out,tmp_path/'summary')
    stats=json.loads((tmp_path/'summary/statistics.json').read_text())
    assert stats['complete'] is False
    assert stats['groups']['all']['unprocessed_or_failed_pairs']==1
    assert stats['groups']['all']['primary_classes']['stable']['percent_of_completed']==100
    assert stats['groups']['all']['primary_classes']['stable']['percent_of_expected']==50
    a.scope='strict'
    with pytest.raises(ValueError,match='Different protocol'):
        asyncio.run(run(a))
