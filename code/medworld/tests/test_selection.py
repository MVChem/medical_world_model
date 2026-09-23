import unittest
from collections import Counter
from medworld.evaluation.selection import (reference_manifest, select_vqa,
                                          segmentation_reference_sha256, validate_references)

class SelectionTest(unittest.TestCase):
    def test_balanced_order_independent_answer_independent(self):
        rows=[{'id':f'{kind}:{i}', 'semantic_type':kind,'answer':['yes']}
              for kind in ('verify','choose','query') for i in range(140)]
        indices,a=select_vqa(rows,100,42)
        self.assertEqual(Counter(rows[i]['semantic_type'] for i in indices),
                         {'verify':100,'choose':100,'query':100})
        _,b=select_vqa([dict(r,answer=['no']) for r in reversed(rows)],100,42)
        self.assertEqual(a,b)
        self.assertEqual(len(set(a['ids'])),300)
        with self.assertRaises(ValueError):select_vqa(rows,200,42)

    def test_reference_manifest_excludes_predictions_and_rejects_changed_targets(self):
        row = {'id': 'c1', 'patient': 'p1', 'labels': [0, 1, -1], 'probabilities': [.1, .9, .5]}
        references = reference_manifest('classification', [row])
        self.assertEqual(references, reference_manifest('classification', [{**row, 'probabilities': [.8, .2, .1]}]))
        summary = {'references': {'classification': references}, 'tasks': {'classification': {'n': 1}}}
        validate_references(summary, 'classification', [row])
        with self.assertRaisesRegex(ValueError, 'IDs or references'):
            validate_references(summary, 'classification', [{**row, 'labels': [1, 1, -1]}])
        with self.assertRaisesRegex(ValueError, 'unique'):
            reference_manifest('classification', [row, row])

    def test_vqa_selection_must_describe_the_actual_evaluated_ids(self):
        row = {'id': 'v1', 'patient': 'p1', 'question': 'Q?', 'answer': ['yes'], 'semantic_type': 'verify'}
        _, selection = select_vqa([row])
        summary = {'references': {'vqa': reference_manifest('vqa', [row])},
                   'tasks': {'vqa': {'n': 1}}, 'vqa_selection': selection, 'limit': None}
        validate_references(summary, 'vqa', [row])
        summary['vqa_selection']['ids'] = ['other']
        with self.assertRaisesRegex(ValueError, 'sampling protocol'):
            validate_references(summary, 'vqa', [row])

    def test_segmentation_requires_mask_digest_and_volume_semantics(self):
        row = {'id': 'slice1', 'patient': 'p1', 'dataset': 'ucsf', 'volume_id': 'vol1',
               'active_channels': [3], 'target_names': ['tumor'], 'target_sha256': 'a' * 64}
        reference = reference_manifest('segmentation', [row])
        for change in ({'target_sha256': 'b' * 64}, {'volume_id': 'vol2'}, {'active_channels': [4]}):
            self.assertNotEqual(reference, reference_manifest('segmentation', [{**row, **change}]))
        del row['target_sha256']
        with self.assertRaisesRegex(ValueError, 'missing test reference field'):
            reference_manifest('segmentation', [row])

    def test_segmentation_fingerprint_distinguishes_equal_area_masks(self):
        import torch
        target = torch.tensor([[[[1., 0.], [0., 0.]]]])
        mask = torch.ones_like(target)
        digest = segmentation_reference_sha256(target, mask)
        self.assertEqual(digest, segmentation_reference_sha256(target.clone(), mask.clone()))
        moved = target.flip(-1)
        self.assertNotEqual(digest, segmentation_reference_sha256(moved, mask))
        self.assertNotEqual(digest, segmentation_reference_sha256(target, mask * 0))
