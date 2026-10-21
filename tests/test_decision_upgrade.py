"""Adversarial checks for the preview outcome-cue changes."""
import unittest
from test_stage44 import document,decision
from stage44_brief import audit_brief
from stage44_attribution import attribution_map

class DecisionUpgradeTests(unittest.TestCase):
    def test_current_punctuated_order_beats_passive_report(self):
        doc=document('1. The petition came to be dismissed by a Division Bench of the High Court.\n'
                     '2. The claimant challenges that order.\n'
                     '3. We find the impugned order unsustainable. It is set aside. The appeal is, accordingly, allowed.\n'
                     '4. The claimant shall receive Rs. 10,000 within four weeks, subject to deduction of payments already made.')
        text,result=decision(doc)
        self.assertIn('is, accordingly, allowed',text)
        self.assertIn('subject to deduction',text)
        self.assertIn('Rs. 10,000',text)
        self.assertNotIn('came to be dismissed',text)
        self.assertTrue(audit_brief(doc,result)['exact_source_check_passed'])

    def test_passive_report_alone_is_not_current_disposition(self):
        for court in ['the High Court','a Division Bench of the High Court','the Trial Court']:
            with self.subTest(court=court):
                doc=document(f'1. The petition came to be dismissed by {court}.\n2. The appellant challenges that decision.')
                text,_=decision(doc)
                self.assertEqual(text,'')
                self.assertEqual(attribution_map(doc)[0]['label'],'reported_material')

    def test_batch_disposition_keeps_adjacent_qualifications(self):
        doc=document('1. Having considered these civil appeals, we find no basis for interference. They are accordingly dismissed. In pending matters, the Tribunal must examine the medical evidence.\n'
                     '2. With these clarifications, the present batch of matters stands dismissed.\n'
                     '3. There shall be no order as to costs.')
        text,result=decision(doc)
        self.assertIn('Tribunal must examine',text)
        self.assertIn('present batch of matters stands dismissed',text)
        self.assertIn('no order as to costs',text)
        self.assertTrue(audit_brief(doc,result)['exact_source_check_passed'])

    def test_punctuation_does_not_override_negation_or_submissions(self):
        for text in ['The appeal is, accordingly, not allowed.',
                     'The petitioner submits that the appeal is, accordingly, allowed.',
                     'Whether the appeal is, accordingly, allowed?',
                     'The respondent argued that the present batch of matters stands dismissed.']:
            with self.subTest(text=text):self.assertEqual(decision(document('1. '+text))[0],'')

    def test_reported_order_does_not_hide_explicit_current_reversal(self):
        doc=document('1. We allow the appeal against the petition dismissed by the High Court.')
        self.assertIn('We allow the appeal',decision(doc)[0])

if __name__=='__main__':unittest.main()
