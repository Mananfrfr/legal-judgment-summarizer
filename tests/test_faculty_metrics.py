import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('faculty_report',Path(__file__).resolve().parents[1]/'scripts/build_faculty_metrics_report.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class FacultyMetricsTests(unittest.TestCase):
    def rows(self):
        return [dict(case_id=c,method='model',**{f'{m}_{k}':value for m in module.METRICS for k in ['precision','recall','f1']}) for c,value in [('a',.2),('b',.8)]]
    def test_aggregate_uses_all_cases(self):
        self.assertEqual(module.aggregate_rows(self.rows(),['a','b'],['model'])['model']['rougeL_f1'],.5)
    def test_incomplete_duplicate_and_nonfinite_rejected(self):
        rows=self.rows()
        for bad in [rows[:1],rows+[rows[0]]]:
            with self.assertRaises(ValueError):module.aggregate_rows(bad,['a','b'],['model'])
        rows[0]['rougeL_f1']=float('nan')
        with self.assertRaises(ValueError):module.aggregate_rows(rows,['a','b'],['model'])
