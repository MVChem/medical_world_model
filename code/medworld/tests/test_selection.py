import unittest
from collections import Counter
from medworld.evaluation.selection import select_vqa

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
