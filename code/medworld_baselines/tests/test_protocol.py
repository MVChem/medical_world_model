import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stats import probability_metrics,qa_metrics
from infer import prompt_for,probability_from_logprobs

class ProtocolTests(unittest.TestCase):
    def test_calibration_known_answer_and_closed_last_bin(self):
        r=probability_metrics([[0],[1]],[[0.],[1.]],['a'])
        self.assertEqual(r['ap'],1);self.assertEqual(r['auroc'],1)
        self.assertEqual(r['brier'],0);self.assertEqual(r['ece'],0)
        self.assertEqual(r['per_finding']['a']['bins'][-1]['n'],1)

    def test_reference_unknown_mask_and_reference_support(self):
        y=[[0,1,-1],[1,1,-2],[-1,1,-1]];p=[[.25,.4,.9],[.75,.4,.1],[1,.4,.8]]
        r=probability_metrics(y,p,['a','b','c'])
        self.assertEqual(r['rank_supported'],['a'])
        self.assertEqual(r['calibration_supported'],['a','b'])
        self.assertAlmostEqual(r['per_finding']['a']['brier'],.0625)
        self.assertAlmostEqual(r['per_finding']['a']['ece'],.25)

    def test_nonfinite_not_silently_dropped(self):
        for value in [np.nan,np.inf,-.1,1.1]:
            with self.assertRaises(ValueError):probability_metrics([[0]],[[value]],['a'])

    def test_missing_qa_output_penalized(self):
        r=qa_metrics([dict(text='',answer=['edema']),dict(text='["edema"]',answer=['edema'])],['edema'])
        self.assertEqual(r['n'],2);self.assertEqual(r['invalid_outputs'],1)
        self.assertEqual(r['exact_match'],.5);self.assertEqual(r['micro_f1'],.5)

    def test_source_only_prompt_invariant(self):
        row=dict(report='current evidence',ehr='past EHR',horizon=1,target_report='secret A',realized_gap_hours=33)
        before=prompt_for('table1_report',row)
        row.update(target_report='secret B',realized_gap_hours=70)
        self.assertEqual(before,prompt_for('table1_report',row))
        self.assertNotIn('secret',before)
        self.assertNotIn('current evidence',prompt_for('table2_report',row))

    def test_raw_likelihood_pair_normalization(self):
        self.assertAlmostEqual(probability_from_logprobs(np.log(.3),np.log(.1)),.75)
        self.assertAlmostEqual(probability_from_logprobs(-100,-100),.5)

if __name__=='__main__':unittest.main()
