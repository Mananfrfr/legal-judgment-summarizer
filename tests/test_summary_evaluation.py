import unittest
from summary_evaluation import aggregate,brief_text,lead_summary,mean_interval,overlap_scores

class SummaryEvaluationTests(unittest.TestCase):
    def test_reference_identity_and_empty_output(self):
        ref='The appeal is dismissed with no order as to costs.'
        self.assertEqual(overlap_scores(ref,ref)['rougeL_f1'],1.0)
        self.assertTrue(all(v==0 for v in overlap_scores(ref,'').values()))

    def test_serialization_omits_audit_and_preserves_app_order(self):
        brief={'sections':{k:[{'sentence':k}] for k in ['decision','reasoning','issue','background']},'warnings':['not a summary']}
        self.assertEqual(brief_text(brief),'background\nissue\nreasoning\ndecision')

    def test_baseline_stops_at_first_overflow_without_cherry_picking(self):
        sentences=[{'sentence':s} for s in ['one two','three four five','six']]
        self.assertEqual(lead_summary(sentences,4),'one two')
        self.assertEqual(lead_summary(sentences,1),'one two')

    def test_bootstrap_constant_and_reproducible(self):
        result=mean_interval([.25]*10)
        self.assertEqual(result['ci95_low'],.25);self.assertEqual(result['ci95_high'],.25)
        self.assertEqual(mean_interval([0,.1,.8,1]),mean_interval([0,.1,.8,1]))

    def test_failures_stay_in_denominator_and_missing_rows_fail(self):
        good=dict(case_id='a',method='app',status='ok',summary_words=10,budget=800,**overlap_scores('appeal allowed','appeal allowed'))
        failed=dict(case_id='b',method='app',status='failed',summary_words=0,budget=800,**overlap_scores('appeal allowed',''))
        result,_=aggregate([good,failed],['a','b'],['app'])
        self.assertEqual(result['app']['rougeL_f1']['mean'],.5)
        self.assertEqual(result['app']['failures'],1)
        with self.assertRaises(ValueError):aggregate([good],['a','b'],['app'])
        with self.assertRaises(ValueError):aggregate([good,good,failed],['a','b'],['app'])

if __name__=='__main__':unittest.main()
