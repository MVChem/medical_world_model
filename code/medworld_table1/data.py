"""Explicit source and target tensors; all report truncation is fixed in config."""
from pathlib import Path
import hashlib
import numpy as np
import torch

from common import load_rows


class Corpus:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        root = Path(cfg['cache'])
        self.observations = load_rows(root / 'observations.jsonl')
        self.lookup = {r['id']: i for i, r in enumerate(self.observations)}
        self.pairs = {s: load_rows(root / f'{s}.jsonl') for s in ('train', 'validate', 'test')}
        self.features = np.load(root / 'vjepa_features.npy', mmap_mode='r')
        assert len(self.features) == len(self.observations)
        self.tokens = tokenizer([r['report'] for r in self.observations], add_special_tokens=False,
                                truncation=True, max_length=cfg['report_tokens'])['input_ids']
        self.context_tokens = self.tokens
        if cfg.get('use_ehr'):
            self.context_tokens = []
            report_header = tokenizer.encode('Current radiograph report:\n', add_special_tokens=False)
            ehr_header = tokenizer.encode('\n\n', add_special_tokens=False)
            for row, report in zip(self.observations, self.tokens):
                # Independent budgets preserve both modalities; report targets
                # and Copy Current still contain only the original report.
                ehr = tokenizer.encode(row['ehr_text'], add_special_tokens=False)
                if len(ehr) > cfg['ehr_tokens']:
                    raise ValueError('EHR serialization exceeded its whole-line budget')
                self.context_tokens.append(report_header + report + ehr_header + ehr)
        self.stage1 = [i for i, r in enumerate(self.observations) if r['split'] == 'train']
        self._permutations = {}
        self.donors = {}
        if cfg.get('state_condition') == 'shuffled':
            # Donors come from the same official split and a different patient;
            # their current evidence is allowed, never their future evidence.
            for split, rows in self.pairs.items():
                for row in rows:
                    position = int(hashlib.sha256(f"{cfg['seed']}:state-donor:{row['id']}".encode()).hexdigest(), 16) % len(rows)
                    for offset in range(len(rows)):
                        donor = rows[(position + offset) % len(rows)]
                        if donor['patient'] != row['patient']:
                            self.donors[row['id']] = donor['source']
                            break
                    else:
                        raise ValueError('Shuffled-state split requires at least two patients')

    def pad(self, sequences, *, left=False):
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), self.tokenizer.pad_token_id or self.tokenizer.eos_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            if seq:
                section = slice(width-len(seq), width) if left else slice(0, len(seq))
                ids[i, section] = torch.tensor(seq)
                mask[i, section] = 1
        return ids, mask

    def batch(self, rows, device='cuda', stage1=False):
        batch = {}
        if stage1:
            rows = [dict(source=self.observations[i]['id'], target=self.observations[i]['id'], horizon=0) for i in rows]
        for side in ('source', 'target'):
            idx = [self.lookup[r[side]] for r in rows]
            tokens = [self.tokens[i] for i in idx]
            batch[side+'_ids'], batch[side+'_mask'] = self.pad([self.context_tokens[i] for i in idx], left=True)
            batch[side+'_target_ids'], batch[side+'_target_mask'] = self.pad([t+[self.tokenizer.eos_token_id] for t in tokens])
            batch[side+'_features'] = torch.from_numpy(np.array(self.features[idx], copy=True))
            batch[side+'_labels'] = torch.tensor([self.observations[i]['labels'] for i in idx])
        if self.donors and not stage1:
            idx = [self.lookup[self.donors[r['id']]] for r in rows]
            batch['donor_ids'], batch['donor_mask'] = self.pad([self.context_tokens[i] for i in idx], left=True)
            batch['donor_features'] = torch.from_numpy(np.array(self.features[idx], copy=True))
        batch['horizon'] = torch.tensor([r['horizon'] for r in rows], dtype=torch.long)
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['_source_images'] = [self.observations[self.lookup[r['source']]]['image'] for r in rows]
        return batch

    def training_batch(self, stage, step, micro):
        # Each optimizer step has a deterministic data stream, shared by matched control.
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg['seed'], stage, step, micro]))
        population = self.stage1 if stage == 1 else self.pairs['train']
        if self.cfg.get('sampling') == 'permutation':
            start = (step*self.cfg['gradient_accumulation'] + micro)*self.cfg['batch_size']
            chosen = []
            for position in range(start, start+self.cfg['batch_size']):
                epoch, offset = divmod(position, len(population))
                key = (stage, epoch)
                if key not in self._permutations:
                    self._permutations = {key: np.random.default_rng(
                        np.random.SeedSequence([self.cfg['seed'], stage, epoch])).permutation(len(population))}
                chosen.append(self._permutations[key][offset])
        else:
            chosen = rng.choice(len(population), self.cfg['batch_size'], replace=False)
        rows = [population[i] for i in chosen]
        return self.batch(rows, stage1=stage == 1)
