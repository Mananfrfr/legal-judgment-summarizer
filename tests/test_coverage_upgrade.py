"""Coverage must improve without turning cited orders into current decisions."""
import unittest
from test_stage44 import document,decision
from stage44_brief import build_brief,audit_brief

class CoverageUpgradeTests(unittest.TestCase):
    def test_quote_after_colon_without_space_remains_reported(self):
        d=document('1. The earlier court said:"We allow this appeal."\n2. We dismiss the present appeal.')
        text,_=decision(d)
        self.assertNotIn('We allow',text)
        self.assertIn('We dismiss',text)

    def test_orphan_closing_quote_does_not_swallow_decision(self):
        d=document('1. The business manufactures steel windows and other goods".\n2. We dismiss the appeal.')
        self.assertFalse(d.sentences[-1]['quoted'])
        self.assertIn('We dismiss',decision(d)[0])

    def test_new_quote_after_missing_close_does_not_invert_state(self):
        d=document('1. The party describes an "income tax refund.\n2. The statute provides "The rebate shall be withheld."\n3. We dismiss the appeal.')
        self.assertTrue(d.sentences[1]['quoted'])
        self.assertFalse(d.sentences[-1]['quoted'])
        self.assertIn('We dismiss',decision(d)[0])

    def test_short_terminal_disposition(self):
        for ending in ['Appeal allowed.','Appeal dismissed.','R.K.P.S. Appeal dismissed.']:
            with self.subTest(ending=ending):
                d=document('1. The claimant challenges the calculation.\n2. '+ending)
                self.assertIn(ending,decision(d)[0])

    def test_accept_appeal_and_modal_conclusion(self):
        for ending in ['We accept the appeal and answer the question in the affirmative.',
                       'Therefore this appeal must be dismissed with costs.']:
            self.assertIn(ending,decision(document('1. The tax calculation is challenged.\n2. '+ending))[0])

    def test_requests_quotes_and_negation_remain_excluded(self):
        for ending in ['The appellant submitted that this appeal must be dismissed.',
                       'The earlier court said: “Appeal allowed.”',
                       'This appeal must not be dismissed.',
                       'We accept the evidence tendered by the appellant.']:
            self.assertEqual(decision(document('1. '+ending))[0],'')

    def test_nonterminal_reporter_line_is_not_an_anchor(self):
        d=document('1. Appeal allowed.\n'+'\n'.join(f'{i}. The remaining questions concern the calculation of tax.' for i in range(2,12)))
        self.assertEqual(decision(d)[0],'')

    def test_coverage_uses_spare_budget_without_rewriting(self):
        facts=['The claimant owned agricultural land near the northern boundary.',
               'An assessment officer valued the property using comparable sales.',
               'The municipality constructed a public road through the village.',
               'Compensation was deposited in a bank account before possession.',
               'A separate notification covered the adjoining residential buildings.',
               'Several witnesses described improvements made during the tenancy.',
               'The original conveyance included access rights over a private pathway.',
               'An independent survey recorded the dimensions of every disputed parcel.']
        d=document('\n'.join(f'{i+1}. {t}' for i,t in enumerate(facts)))
        predictions={s['id']:dict(importance_score=.8,role_code='FAC',role_confidence=.9) for s in d.sentences}
        old=build_brief(d,400);new=build_brief(d,400,predictions)
        self.assertGreater(new['summary_words'],old['summary_words'])
        self.assertLessEqual(new['summary_words'],400)
        self.assertTrue(audit_brief(d,new)['exact_source_check_passed'])
        self.assertEqual(len({e['id'] for v in new['sections'].values() for e in v}),sum(map(len,new['sections'].values())))

    def test_predicted_analysis_does_not_promote_a_letter(self):
        d=document('1. The following letter reads: Dear Sirs, the amount is payable immediately. We consider this demand final.\n2. We dismiss the appeal.')
        p={s['id']:dict(importance_score=.99,role_code='ANALYSIS',role_confidence=.99) for s in d.sentences}
        result=build_brief(d,800,p)
        self.assertFalse(any('Dear Sirs' in e['sentence'] or 'demand final' in e['sentence'] for e in result['sections']['reasoning']))

    def test_model_only_reasoning_is_marked_for_review(self):
        d=document('1. The claimant owned a house.\n2. The property had a large garden.\n3. The tenant paid monthly rent.\n4. The statutory limitation excludes every claim filed after the prescribed period.')
        p={s['id']:dict(importance_score=.5,role_code='ANALYSIS',role_confidence=.9) for s in d.sentences}
        r=build_brief(d,800,p)
        self.assertTrue(r['sections']['reasoning'])
        self.assertTrue(any('predicted roles' in w for w in r['warnings']))
        self.assertTrue(audit_brief(d,r)['exact_source_check_passed'])

    def test_giant_paragraph_does_not_become_entire_decision(self):
        history=' '.join('The record describes the ownership and subsequent assessment of agricultural property.' for _ in range(90))
        d=document('1. '+history+' We allow the appeal. The authority shall refund Rs. 500 within four weeks, subject to adjustment of prior payments.')
        text,r=decision(d)
        self.assertIn('We allow the appeal',text)
        self.assertIn('subject to adjustment',text)
        self.assertLess(len(text.split()),100)
        self.assertTrue(any('Oversized source paragraph' in w for w in r['warnings']))

if __name__=='__main__':unittest.main()
