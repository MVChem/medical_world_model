import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build import run,LABEL_COLUMNS
from common import read_jsonl
from qc_images import inspect_image


def table(path,fields,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open if path.suffix=='.gz' else open)(path,'wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def fixture(tmp_path,views):
    cxr,iv=tmp_path/'cxr',tmp_path/'iv'
    p='10000001';metadata=[];split=[];labels=[]
    for i,view in enumerate(views,1):
        study=str(i);im='image'+str(i)
        directory=cxr/'files/p10'/('p'+p)/('s'+study)
        directory.mkdir(parents=True)
        Image.fromarray(np.random.default_rng(i).integers(0,256,size=(40,40),dtype=np.uint8)).save(directory/(im+'.jpg'))
        (directory.parent/('s'+study+'.txt')).write_text('FINDINGS AND IMPRESSION: Possible effusion. No pneumothorax.')
        metadata.append(dict(subject_id=p,study_id=study,dicom_id=im,ViewPosition=view,
            StudyDate=f'2100010{i}',StudyTime='120000.500',Rows='40',Columns='40',
            PerformedProcedureStepDescription='PORTABLE AP',PatientOrientationCodeSequence_CodeMeaning='Erect'))
        split.append(dict(subject_id=p,study_id=study,dicom_id=im,split='train'))
        if i<3:labels.append(dict(subject_id=p,study_id=study,**{k:'' for k in LABEL_COLUMNS}))
    table(cxr/'mimic-cxr-2.0.0-metadata.csv',list(metadata[0]),metadata)
    table(cxr/'mimic-cxr-2.0.0-split.csv',list(split[0]),split)
    table(cxr/'mimic-cxr-2.0.0-chexpert.csv',['subject_id','study_id',*LABEL_COLUMNS],labels)
    table(iv/'hosp/patients.csv.gz',['subject_id','gender'],[dict(subject_id=p,gender='F')])
    table(iv/'hosp/admissions.csv.gz',['subject_id','hadm_id','admittime','dischtime','edregtime'],[
        dict(subject_id=p,hadm_id='11',admittime='2100-01-01 13:00:00',dischtime='2100-01-05 00:00:00',edregtime='2100-01-01 10:00:00')])
    table(iv/'icu/icustays.csv.gz',['subject_id','hadm_id','stay_id','intime','outtime'],[])
    return argparse.Namespace(cxr=cxr,iv=iv,out=tmp_path/'out',min_gap_hours=6,max_gap_hours=720)


def test_full_inventory_report_cleaning_ed_link_and_missing_labels(tmp_path):
    args=fixture(tmp_path,['AP','AP','AP'])
    run(args)
    pairs=list(read_jsonl(args.out/'pairs.jsonl'))
    assert len(pairs)==2
    assert all(r['tiers']['same_admission_6h_72h'] for r in pairs)
    assert not pairs[0]['same_strict_admission']
    assert pairs[1]['audit_flags']['missing_chexpert_record']
    assert pairs[1]['audit_flags']['known_future_finding_count']==0
    reports=list(read_jsonl(args.out/'studies.jsonl'))
    assert all(r['report']['text'].count('Possible effusion.')==1 for r in reports)
    assert reports[0]['timestamp'].endswith('.500000')
    run(args)  # Completed-input signature supports restart.


def test_pairing_never_skips_an_intervening_lateral_only_study(tmp_path):
    args=fixture(tmp_path,['AP','LATERAL','AP'])
    run(args)
    assert list(read_jsonl(args.out/'pairs.jsonl'))==[]
    rejected=list(read_jsonl(args.out/'rejected_adjacent_pairs.jsonl'))
    assert len(rejected)==2
    assert all('no_common_frontal_view' in r['reasons'] for r in rejected)
    assert len(list(read_jsonl(args.out/'images.jsonl')))==3


def test_image_qc_detects_corruption_and_hashes_decoded_pixels(tmp_path):
    image=tmp_path/'valid.jpg'
    Image.fromarray(np.arange(1600,dtype=np.uint8).reshape(40,40)).save(image)
    record=dict(dicom_id='a',path=str(image),rows='40',columns='40')
    qc=inspect_image(record)
    assert qc['valid'] and qc['dimensions_match_metadata']
    assert len(qc['pixel_sha256'])==64 and len(qc['dhash64'])==16
    image.write_bytes(b'not an image')
    assert not inspect_image(record)['valid']
