"""Freeze identical held-out inputs before inspecting any model results."""
import argparse
import collections
import os
from pathlib import Path
from base import *

def main(run):
    os.umask(0o077)
    if (run/'protocol.json').exists():
        raise FileExistsError('Use the existing frozen cohort or a fresh run directory')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(T1/'weights/Qwen3.5-0.8B', local_files_only=True)
    cache1 = T1/'data/linked_20260909'
    cache2 = T2/'data/overnight_20260910'
    qa_root = T2/'data/slot44_20260911_derived_v2'
    obs = {r['id']:r for r in rows(cache1/'observations.jsonl')}
    selection = read(qa_root/'selection.json')
    overrides = selection['patient_split_overrides']
    all2 = rows(cache2/'observations.jsonl')
    by2 = {r['id']:r for r in all2}
    vocabulary = read(qa_root/'answer_vocabulary.json')['labels']
    def cap(text, n=384):
        return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:n], skip_special_tokens=True)
    counts = {}
    for split in ['test', 'validate']:
        inputs1, refs1 = [], []
        for pair in rows(cache1/f'{split}.jsonl'):
            source, target = obs[pair['source']], obs[pair['target']]
            # Deliberately disjoint input/target artifacts. No realized gap or target IDs in prompts.
            inputs1.append(dict(id=pair['id'], image=source['image'], report=cap(source['report']),
                ehr=source['ehr_text'], horizon=pair['horizon'], patient=pair['patient']))
            refs1.append(dict(id=pair['id'], patient=pair['patient'], current_report=source['report'],
                target_report=target['report'], horizon=pair['horizon'], source=pair['source'], target=pair['target']))
        selected2 = [r for r in all2 if overrides.get(str(r['subject_id']),r['split']) == split]
        inputs2 = [dict(id=r['id'], image=r['image'], patient=str(r['subject_id']),
            classification=bool(r['tasks']['classification']), report_generation=bool(r['report_valid'] and r['report'].strip()))
            for r in selected2 if r['tasks']['classification'] or (r['report_valid'] and r['report'].strip())]
        refs2 = [dict(id=r['id'], patient=str(r['subject_id']), labels=r['labels'], report=r['report']) for r in selected2]
        qa = [r for r in rows(qa_root/'disease_questions.jsonl') if r['split']==split]
        inputsq = [dict(id=q['id'], image=by2[q['image_id']]['image'], image_id=q['image_id'],
            patient=q['subject_id'], question=q['question']) for q in qa]
        for name, values in [('table1_inputs',inputs1),('table1_references',refs1),('table2_inputs',inputs2),
                             ('table2_references',refs2),('qa_inputs',inputsq),('qa_references',qa)]:
            write_rows(run/'cohort'/f'{name}_{split}.jsonl', values)
        counts[split] = dict(table1=len(inputs1), table1_patients=len({r['patient'] for r in inputs1}),
            classification=sum(r['classification'] for r in inputs2), report_generation=sum(r['report_generation'] for r in inputs2),
            table2_patients=len({r['patient'] for r in inputs2}), derived_qa=len(qa), derived_qa_images=len({q['image_id'] for q in qa}))
    atomic(run/'vocabulary.json', vocabulary)
    inventory = models()
    for m in inventory:
        root=Path(m['path'])
        m['config_sha256']=digest(root/'config.json')
        weights=list(root.glob('*.safetensors'))
        if not weights or any(not p.exists() for p in weights):
            raise FileNotFoundError(m['id'])
        m['weight_files']=[dict(name=p.name,bytes=p.stat().st_size,resolved=str(p.resolve())) for p in weights]
    atomic(run/'models.json', inventory)
    protocol=dict(version=1,created='2026-09-11',deadline='2026-09-12T08:00:00+08:00',training='none; original local public checkpoints',
        counts=counts,seed=20260911,image='512 x 512 aspect-preserving bicubic resize and black pad; native model processor follows',
        table1_input='source image + source report capped to 384 common Qwen tokens + original source EHR + requested horizon; no future evidence',
        table2_input='image only; question additionally supplied for QA; same-exam report withheld from every task',
        probabilities='P(Yes)/(P(Yes)+P(No)) from unmodified next-token log probabilities; Yes and No must each be a single complete token in each tokenizer; no EOS included; no generated percentages',
        probability_threshold=0.5, calibration='raw probabilities; no test fitting',ece_bins=10,
        macro_support='AP/AUROC: findings with both reference classes; Brier/ECE: findings with >=1 known reference; fixed reference-only masks',
        transition='CheXbert current/reference and generated future report; onset/resolution macro over reference-supported events; probability-threshold alternative reported separately',
        direction='unavailable: disease/laterality direction references have not passed adjudication; no regex or Qwen pseudo-truth substitution',
        green='official frozen GREEN evaluator; status tracked separately',
        vqa='Official MIMIC-CXR-VQA unavailable locally. Separate positive-answer-only Chest ImaGenome derived QA pilot; not the official VQA table cells',
        grounding='MS-CXR phrase/box benchmark unavailable locally; anatomical boxes are not substituted for lesion phrase grounding',
        segmentation_sr='not applicable to these native text-output VLM interfaces; no trained pixel decoder',
        public_pretraining_overlap='unknown; local patient-disjoint splits do not rule out public model pretraining overlap',
        generation=dict(temperature=0,max_report_tokens=384,max_qa_tokens=192,qwen_thinking=False),
        file_sha256={p.name:digest(p) for p in sorted((run/'cohort').glob('*'))},
        sources={str(p):digest(p) for p in [cache1/'test.jsonl',cache1/'validate.jsonl',cache1/'observations.jsonl',cache2/'observations.jsonl',qa_root/'selection.json',qa_root/'disease_questions.jsonl']})
    atomic(run/'protocol.json',protocol)
    print(counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
