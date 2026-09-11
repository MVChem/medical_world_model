"""Build reviewable QA manifests without modifying the original observation caches."""
import argparse
import collections
import csv
import json
import os
import zipfile
from pathlib import Path
from bootstrap import ROOT, atomic_json, load_rows, write_rows, digest
from slot44_evaluation import canonical

IMAGENOME=Path('/home/data1/data/MIMIC/MIMIC_CXR/chest-imagenome-dataset-1.0.0')
CATEGORIES={'anatomicalfinding','disease'}


def main(args):
    os.umask(0o077)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.json').exists():
        raise FileExistsError('Selection is frozen; use a new output directory')
    observations=load_rows(args.cache/'observations.jsonl')
    by_image={r['id']:i for i,r in enumerate(observations)}
    gold_file=IMAGENOME/'gold_dataset/gold_attributes_relations_500pts_500studies1st.txt'
    gold=list(csv.DictReader(gold_file.open(),delimiter='\t'))
    gold_patients={str(r['patient_id']) for r in gold}
    labels=[]
    for line in (IMAGENOME/'semantics/attribute_relations_v1.txt').read_text().splitlines():
        parts=line.strip().rstrip(',').split('|')
        if len(parts)==3 and parts[0] in CATEGORIES:
            labels.append(canonical(parts[2]))
    labels=sorted(set(labels))
    overrides={p:'test' for p in gold_patients}
    counts=collections.Counter();qas=[];sources={'gold_attributes':digest(gold_file)}

    if args.vqa_dir:
        raw=[]
        for split,name in [('train','train.json'),('validate','valid.json'),('test','test.json')]:
            path=args.vqa_dir/name
            rows=json.loads(path.read_text());assert isinstance(rows,list)
            sources[name]=digest(path)
            for q in rows:q=dict(q,source_split=split);raw.append(q)
        gold_patients|={str(q['subject_id']) for q in raw if q['source_split']=='test'}
        for q in raw:
            pid=str(q['subject_id'])
            if q['source_split']=='validate' and pid not in gold_patients:
                overrides[pid]='validate'
        overrides.update({p:'test' for p in gold_patients})
        # Existing official test patients always remain held out, even if VQA calls them silver-valid.
        overrides.update({str(r['subject_id']):'test' for r in observations if r['split']=='test'})
        for q in raw:
            counts['official_raw']+=1
            if q.get('semantic_type')!='query' or q.get('content_type')!='attribute':
                counts['excluded_question_type']+=1;continue
            arguments=q.get('template_arguments',{})
            cats=arguments.get('category',{})
            cats=list(cats.values()) if isinstance(cats,dict) else cats
            if not cats or any(str(c).replace(' ','').lower() not in CATEGORIES for c in cats):
                counts['excluded_non_disease_category']+=1;continue
            if any(arguments.get(k) for k in ['gender','viewpos']):
                counts['excluded_non_image_context']+=1;continue
            i=by_image.get(q['image_id'])
            if i is None:counts['image_not_in_cached_subset']+=1;continue
            r=observations[i]
            if not r['report_valid'] or not r['report'].strip():
                counts['missing_valid_report']+=1;continue
            assert str(q['subject_id'])==str(r['subject_id']) and str(q['study_id'])==str(r['study_id'])
            if not isinstance(q.get('answer'),list):
                counts['missing_answer']+=1;continue
            answer=sorted({canonical(a) for a in q['answer']})
            if set(answer)-set(labels):counts['out_of_ontology_answer']+=1;continue
            qas.append(dict(id=f"official_{q['source_split']}_{q['idx']}", image_index=i,image_id=r['id'],
                subject_id=str(r['subject_id']),question=q['question'],answer=answer,
                subset='region' if arguments.get('object') else 'whole',annotation='gold' if q['source_split']=='test' else 'silver',
                source_split=q['source_split'],template_program=q.get('template_program')))
        source='official MIMIC-Ext-MIMIC-CXR-VQA 1.0.0, selected disease/finding-list questions'
    elif args.imagenome_derived:
        # This alternative is separately named and must never be reported as official VQA.
        gold_by_image=collections.defaultdict(list)
        for r in gold:gold_by_image[Path(r['image_id']).stem].append(r)
        archive=IMAGENOME/'silver_dataset/scene_graph.zip'
        with zipfile.ZipFile(archive) as z:
            names={Path(n).stem.removesuffix('_SceneGraph'):n for n in z.namelist() if n.endswith('.json')}
            for i,r in enumerate(observations):
                if r['view'] not in ('AP','PA') or not r['report_valid'] or not r['report'].strip():continue
                groups=collections.defaultdict(set)
                annotation='silver'
                if str(r['subject_id']) in gold_patients:
                    if r['id'] not in gold_by_image:
                        counts['gold_patient_without_matched_gold_image']+=1;continue
                    annotation='gold'
                    for a in gold_by_image[r['id']]:
                        if a['categoryID'] in CATEGORIES and a['relation']=='1' and a['context']=='yes':
                            groups[a['bbox']].add(canonical(a['label_name']))
                else:
                    name=names.get(r['id'])
                    if name is None:counts['missing_scene_graph']+=1;continue
                    g=json.loads(z.read(name))
                    assert str(g['patient_id'])==str(r['subject_id']) and str(g['study_id'])==str(r['study_id'])
                    for a in g['attributes']:
                        for sentence in a['attributes']:
                            for attribute in sentence:
                                bits=attribute.split('|')
                                if len(bits)==3 and bits[0] in CATEGORIES and bits[1]=='yes':
                                    groups[a['bbox_name']].add(canonical(bits[2]))
                all_labels=set().union(*groups.values()) if groups else set()
                if not all_labels:
                    # Missing/non-affirmed labels are not silently treated as a complete negative answer.
                    counts['no_affirmed_findings_excluded']+=1;continue
                entries=[('whole','List the diseases and anatomical findings described as present in this chest radiograph.',all_labels)]
                entries.extend(('region',f'List the diseases and anatomical findings described as present in the {region}.',a)
                               for region,a in sorted(groups.items()) if a)
                for j,(subset,question,answer) in enumerate(entries):
                    if set(answer)-set(labels):counts['out_of_ontology_answer']+=1;continue
                    qas.append(dict(id=f'derived_{r["id"]}_{j}',image_index=i,image_id=r['id'],subject_id=str(r['subject_id']),
                        question=question,answer=sorted(answer),subset=subset,annotation=annotation))
                if i%2000==0:print('QA preparation images',i,'questions',len(qas),flush=True)
        source='Locally derived Chest ImaGenome affirmed disease/finding lists; NOT the official MIMIC-CXR-VQA benchmark'
        sources['scene_graph_archive_size']=archive.stat().st_size
    else:
        raise ValueError('Provide --vqa-dir or explicitly select --imagenome-derived')

    for q in qas:
        r=observations[q['image_index']]
        q['split']=overrides.get(str(r['subject_id']),r['split'])
        if q['split']=='test' and q['annotation']!='gold':
            # Never call silver labels a gold diagnosis test.
            q['exclude_silver_test']=True
    counts['silver_test_questions_excluded']=sum(q.get('exclude_silver_test',False) for q in qas)
    qas=[q for q in qas if not q.get('exclude_silver_test',False)]
    coverage={s:dict(questions=sum(q['split']==s for q in qas),
                    images=len({q['image_id'] for q in qas if q['split']==s}),
                    patients=len({q['subject_id'] for q in qas if q['split']==s}),
                    whole=sum(q['split']==s and q['subset']=='whole' for q in qas),
                    region=sum(q['split']==s and q['subset']=='region' for q in qas),
                    empty_answers=sum(q['split']==s and not q['answer'] for q in qas)) for s in ['train','validate','test']}
    train_patients={str(r['subject_id']) for r in observations if overrides.get(str(r['subject_id']),r['split'])=='train'}
    assert not train_patients&gold_patients
    excluded=[r for r in observations if r['split']=='train' and overrides.get(str(r['subject_id']),'train')!='train']
    selection=dict(qa_source=source,patient_split_overrides=overrides,gold_test_patients=sorted(gold_patients),
                   observations_sha256=digest(args.cache/'observations.jsonl'),qa_coverage=coverage,counts=dict(counts),
                   removed_from_original_train_images=len(excluded),removed_from_original_train_patients=len({r['subject_id'] for r in excluded}),
                   all_tasks_train_gold_patient_overlap=0,source_hashes=sources,
                   derived_positive_only=bool(args.imagenome_derived),initialization='original pretrained backbones, fresh trainable parameters')
    write_rows(out/'disease_questions.jsonl',qas)
    atomic_json(out/'answer_vocabulary.json',dict(labels=labels,normalization='lowercase and collapsed whitespace only; no semantic merging',
                source='Chest ImaGenome fixed disease/anatomicalfinding ontology'))
    atomic_json(out/'selection.json',selection)
    print(json.dumps(dict(qa_source=source,coverage=coverage,excluded_train_images=len(excluded),counts=dict(counts)),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT/'data/overnight_20260910')
    p.add_argument('--out',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--vqa-dir',type=Path);g.add_argument('--imagenome-derived',action='store_true')
    main(p.parse_args())
