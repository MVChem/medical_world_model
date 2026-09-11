"""Verify the whole planned sample stream and all controlled settings."""
import copy
import ast
import hashlib
import json
from pathlib import Path
from bootstrap import digest
import medworld_common


def shared_model_contract(reference_source):
    """Verify moved definitions against old or new reference snapshots."""
    current = Path(medworld_common.__file__).parent / 'qwen.py'
    reference = reference_source / 'medworld_common/qwen.py'
    if reference.exists():
        assert digest(current) == digest(reference), 'Shared Qwen implementation changed'
        return {'qwen.py': digest(current)}
    reference = reference_source / 'model.py'
    def definitions(path):
        result = {}
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                result[node.name] = ast.dump(node, include_attributes=False)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        result[target.id] = ast.dump(node, include_attributes=False)
        return result
    actual, expected = definitions(current), definitions(reference)
    for name, definition in actual.items():
        assert definition == expected.get(name), f'Shared Qwen definition changed: {name}'
    return dict(reference_sha256=digest(reference), shared_sha256=digest(current),
                identical_definitions=sorted(actual))


CONTROLLED = ['seed','qwen','cache','visual_grid','scale','report_tokens','query_tokens','target_tokens','generation_tokens',
              'lora_rank','lora_alpha','tasks','batch_sizes','gradient_accumulation','learning_rate','lora_learning_rate',
              'warmup_steps','loss_weights','validation_every','validation_count','max_steps','selection','qa_manifest','answer_vocabulary']


def usage(corpus):
    return dict(truncation=corpus.truncation,pools={f'{s}/{t}':len(p) for (s,t),p in corpus.pools.items()},
                pos_weight=corpus.pos_weight.tolist(),classification_label_coverage=corpus.label_coverage,
                missing_report_by_task=corpus.missing_report)


def audit(cfg, corpus, source):
    reference = Path(cfg['reference_run'])
    refcfg = json.loads((reference/'config.json').read_text())
    for key in CONTROLLED:
        assert cfg[key] == refcfg[key], f'Controlled setting differs: {key}'
    refsource = reference.parent/'source'
    identical_files = ['corpus.py','slot44_corpus.py','cache.py','slot44_evaluation.py','evaluation.py','networks.py','slot44_networks.py']
    shared = shared_model_contract(refsource)
    files = {}
    for name in identical_files:
        assert digest(source/name) == digest(refsource/name), f'Reference code changed: {name}'
        files[name] = digest(source/name)
    assert usage(corpus) == json.loads((reference/'data_usage.json').read_text()), 'Task pools, labels or truncation differ'
    ref = copy.copy(corpus)
    ref.cfg = refcfg
    ref.permutations = {}
    corpus.permutations = {}
    fingerprint = hashlib.sha256()
    examples = {t:0 for t in cfg['tasks']}
    updates = {t:0 for t in cfg['tasks']}
    for step in range(cfg['max_steps']):
        task = cfg['tasks'][step % len(cfg['tasks'])]
        ts = step // len(cfg['tasks'])
        updates[task] += 1
        for micro in range(cfg['gradient_accumulation']):
            a = corpus.sample(task,ts,micro)
            b = ref.sample(task,ts,micro)
            assert a == b, f'Sampling differs at step {step}, microbatch {micro}'
            assert len(a) == cfg['batch_sizes'][task]
            examples[task] += len(a)
            fingerprint.update(json.dumps([step,task,micro,a],separators=(',',':')).encode())
    corpus.permutations = {}
    return dict(controlled_settings={k:cfg[k] for k in CONTROLLED},identical_reference_source_hashes=files,shared_model_contract=shared,
                data_usage_identical=True,all_planned_microbatches_identical=True,checked_optimizer_steps=cfg['max_steps'],
                checked_microbatches=cfg['max_steps']*cfg['gradient_accumulation'],sample_stream_sha256=fingerprint.hexdigest(),
                task_updates=updates,task_examples=examples,
                comparison='Step and example counts matched; full-token context may have a different FLOP/time cost')
