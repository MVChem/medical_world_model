"""Frozen image caches, globally disjoint patients, separate disease-list records."""
import json
from pathlib import Path
import numpy as np
import torch
from corpus import Corpus
from bootstrap import FINDINGS, load_rows


SYSTEM = ('Answer the question using the supplied chest radiograph clinical state. '
          'Return only a JSON array of standardized disease or finding names. '
          'Use [] only when the requested set is empty. Do not write a report or explanation.')


class Slot44Corpus(Corpus):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg, tokenizer)
        selection = json.loads(Path(cfg['selection']).read_text())
        overrides = selection['patient_split_overrides']
        self.selection = selection
        for row in self.rows:
            row['split'] = overrides.get(str(row['subject_id']), row['split'])
        self.pools = {(s,t): [i for i,r in enumerate(self.rows) if r['split']==s and r['tasks'][t]
                       and (t!='segmentation' or self.seg_valid[i])]
                      for s in ['train','validate','test'] for t in cfg['tasks']}
        self.qa = []
        self.qa_queries = []
        self.qa_targets = []
        self.qa_by_image = {}
        rejected = {'query_over_budget':0, 'answer_over_budget':0}
        lengths = {'query':[], 'answer':[]}
        for q in load_rows(Path(cfg['qa_manifest'])):
            i = q['image_index']
            assert q['image_id'] == self.rows[i]['id']
            assert q['subject_id'] == str(self.rows[i]['subject_id'])
            assert q['split'] == self.rows[i]['split']
            assert self.rows[i]['report_valid'] and self.rows[i]['report'].strip()
            assert isinstance(q['answer'], list) and all(isinstance(a,str) for a in q['answer'])
            question_tokens = tokenizer(q['question'], add_special_tokens=False)['input_ids']
            target = tokenizer(json.dumps(sorted(set(q['answer'])), ensure_ascii=False), add_special_tokens=False)['input_ids']
            target += [tokenizer.eos_token_id]
            lengths['query'].append(len(question_tokens)); lengths['answer'].append(len(target))
            if len(question_tokens)>cfg['query_tokens']:
                rejected['query_over_budget']+=1; continue
            if len(target)>cfg['target_tokens']:
                rejected['answer_over_budget']+=1; continue
            prompt = tokenizer.apply_chat_template([
                {'role':'system','content':SYSTEM}, {'role':'user','content':q['question']}],
                tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
            qi = len(self.qa)
            self.qa.append(q); self.qa_queries.append(prompt); self.qa_targets.append(target)
            self.qa_by_image.setdefault(i, []).append(qi)
        for s in ['train','validate','test']:
            self.pools[s,'diagnosis'] = [j for j,q in enumerate(self.qa) if q['split']==s]
        for task in cfg['tasks']:
            assert self.pools['train',task], f'No training data for {task}'
            assert self.pools['validate',task], f'No validation data for {task}'
        sets = [{str(r['subject_id']) for r in self.rows if r['split']==s} for s in ['train','validate','test']]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        assert not sets[0] & set(selection['gold_test_patients'])
        self.diagnosis_train_images = sorted({self.qa[q]['image_index'] for q in self.pools['train','diagnosis']})
        self.permutations = {}
        labels = np.array([self.rows[i]['labels'] for i in self.pools['train','classification']])
        self.pos_weight = torch.tensor(np.clip((labels==0).sum(0)/np.maximum((labels==1).sum(0),1),.25,10), dtype=torch.float32)
        self.label_coverage = {}
        for split in ['train','validate','test']:
            a = np.asarray([self.rows[i]['labels'] for i in self.pools[split,'classification']]).reshape(-1,13)
            self.label_coverage[split] = {name:{str(v):int((a[:,j]==v).sum()) for v in [-2,-1,0,1]}
                                          for j,name in enumerate(FINDINGS)}
        self.missing_report = {f'{s}/{t}':sum(not self.rows[i]['report'].strip() for i in pool)
                              for (s,t),pool in self.pools.items() if t!='diagnosis'}
        self.missing_report.update({f'{s}/diagnosis':0 for s in ['train','validate','test']})
        self.truncation = {'report_input': {s:sum(len(tokenizer(r['report'],add_special_tokens=False)['input_ids'])>cfg['report_tokens']
                               for r in self.rows if r['split']==s) for s in ['train','validate','test']},
                           'qa_excluded':rejected,
                           'qa_lengths':{k:{'max':max(v,default=0), 'p99':float(np.percentile(v,99)) if v else None}
                                         for k,v in lengths.items()},
                           'qa_target_truncation':0, 'qa_query_truncation':0}

    def sample(self, task, task_step, micro):
        if task!='diagnosis':
            return super().sample(task, task_step, micro)
        # Sample unique images within a microbatch and rotate their questions by epoch.
        pool = self.diagnosis_train_images
        batch = self.cfg['batch_sizes'][task]
        start = (task_step*self.cfg['gradient_accumulation']+micro)*batch
        epoch, offset = divmod(start, len(pool))
        key = ('qa_images',epoch)
        if key not in self.permutations:
            self.permutations[key] = np.random.default_rng(np.random.SeedSequence([self.cfg['seed'],91,epoch])).permutation(pool)
        ordered = self.permutations[key]
        images = [int(ordered[(offset+j)%len(pool)]) for j in range(min(batch,len(pool)))]
        assert len(images)==len(set(images))
        return [self.qa_by_image[i][(epoch+i)%len(self.qa_by_image[i])] for i in images]

    def batch(self, indices, task, device='cuda'):
        if task!='diagnosis':
            return super().batch(indices, task, device)
        images = [self.qa[q]['image_index'] for q in indices]
        b = super().batch(images, task, device)
        query, qm = self.pad([self.qa_queries[q] for q in indices], left=True)
        target, tm = self.pad([self.qa_targets[q] for q in indices])
        b.update(query_ids=query.to(device), query_mask=qm.to(device),
                 target_ids=target.to(device), target_mask=tm.to(device))
        return b
