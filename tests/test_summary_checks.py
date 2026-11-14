import unittest
from copy import deepcopy
from test_stage44 import document
from summary_checks import check_summary,extract_values,checks_markdown
from stage44_brief import build_brief,audit_brief,summarize_file

def selected(doc, ids):
    return dict(source_sha256=doc.sha256,sections={'decision':[deepcopy(doc.sentences[i]) for i in ids]})

class DetailCheckTests(unittest.TestCase):
    def setUp(self):
        self.doc=document('1. The appeal is allowed.\n2. We direct the insurer to pay Rs. 2,00,000 within six weeks from today, subject to adjustment of earlier payments.')

    def test_linked_outcome_amount_deadline_match(self):
        r=check_summary(self.doc,selected(self.doc,[0,1]))
        self.assertEqual(r['status'],'no_rule_issue_detected')
        self.assertEqual({c['kind'] for c in r['claims']},{'outcome','amount','deadline'})
        self.assertTrue(all(c['status']=='matched_linked_source' for c in r['claims']))
        self.assertFalse(r['legal_correctness_verified'])

    def test_wrong_outcome_amount_deadline_each_detected(self):
        for before,after,index in [('allowed','dismissed',0),('2,00,000','9,00,000',1),('six weeks','nine weeks',1)]:
            with self.subTest(before=before):
                brief=selected(self.doc,[0,1]);brief['sections']['decision'][index]['sentence']=brief['sections']['decision'][index]['sentence'].replace(before,after)
                r=check_summary(self.doc,brief)
                self.assertIn('DECISION_SOURCE_MISMATCH',r['flags'])
                self.assertTrue(any(c['status']=='linked_value_absent' for c in r['claims']))

    def test_amount_elsewhere_cannot_validate_wrong_link(self):
        doc=document('1. We direct payment of Rs. 100.\n2. We direct a separate payment of Rs. 900.')
        brief=selected(doc,[0]);brief['sections']['decision'][0]['sentence']='We direct payment of Rs. 900.'
        r=check_summary(doc,brief)
        self.assertEqual(r['claims'][0]['status'],'linked_value_absent')

    def test_omitted_details_link_back_to_source(self):
        r=check_summary(self.doc,selected(self.doc,[0]))
        self.assertEqual({v['kind'] for v in r['possible_omissions']},{'amount','deadline'})
        self.assertTrue(all(v['source_id']==1 and v['page']==1 for v in r['possible_omissions']))

    def test_dropped_condition_is_not_a_clean_match(self):
        brief=selected(self.doc,[0,1]);brief['sections']['decision'][1]['sentence']=brief['sections']['decision'][1]['sentence'].split(', subject')[0]+'.'
        r=check_summary(self.doc,brief)
        self.assertIn('DECISION_SOURCE_MISMATCH',r['flags'])
        self.assertTrue(any(c['status']=='context_requires_review' for c in r['claims']))

    def test_reported_quoted_submissions_and_hypotheticals_not_source_orders(self):
        for text in ['The High Court dismissed the appeal.', 'The petitioner submitted that the appeal is allowed.',
                     'The previous court said: "The appeal is allowed."', 'If the appeal is allowed, the respondent shall pay Rs. 100.',
                     'Whether the appeal is dismissed?']:
            with self.subTest(text=text):
                doc=document('1. '+text);r=check_summary(doc,selected(doc,[0]))
                self.assertEqual(r['counts']['outcome']['source_candidate_count'],0)
                self.assertIn('DECISION_ATTRIBUTION_REVIEW',r['flags'])

    def test_multiple_outcomes_require_review(self):
        doc=document('1. The appeal is allowed.\n2. The petition is dismissed.')
        self.assertIn('MULTIPLE_SOURCE_OUTCOMES_REVIEW',check_summary(doc,selected(doc,[0,1]))['flags'])

    def test_negation_and_partial_outcome_not_equivalent(self):
        for text,want in [('The appeal is not allowed.','not allowed'),('The appeal is partly allowed.','partly allowed')]:
            self.assertEqual(extract_values(text)[0]['value'],want)

    def test_currency_normalization_retains_scale_and_decimal(self):
        for text,want in [('Rs. 2,00,000','INR 200000'),('INR 2 lakh','INR 200000'),('₹ 1.50 crore','INR 15000000'),('rupees 20.50','INR 20.5'),('Rs 100 thousand','INR 100000')]:
            self.assertEqual(extract_values(text)[0]['value'],want)
        self.assertEqual(extract_values('The owners 100 and others 500 are registry identifiers.'),[])

    def test_time_units_and_business_days_stay_distinct(self):
        self.assertEqual(extract_values('within six weeks')[0]['value'],'6 week')
        self.assertEqual(extract_values('within 6 weeks')[0]['value'],'6 week')
        self.assertNotEqual(extract_values('within 6 business days')[0]['value'],extract_values('within 6 days')[0]['value'])
        self.assertEqual(extract_values('on or before 15 November 2026')[0]['value'],'15 november 2026')

    def test_no_decision_or_unrecognized_outcome_is_not_a_pass(self):
        doc=document('1. We remit this matter for reconsideration.')
        self.assertIn('SUMMARY_OUTCOME_NOT_RECOGNIZED',check_summary(doc,selected(doc,[0]))['flags'])
        self.assertIn('DECISION_MISSING',check_summary(doc,selected(doc,[]))['flags'])

    def test_wrong_document_identity(self):
        brief=selected(self.doc,[0]);brief['source_sha256']='wrong'
        report=check_summary(self.doc,brief)
        self.assertIn('DOCUMENT_IDENTITY_MISMATCH',report['flags'])
        self.assertFalse(any(c['status']=='matched_linked_source' for c in report['claims']))

    def test_audit_and_markdown_integration(self):
        brief=build_brief(self.doc);audit=audit_brief(self.doc,brief)
        self.assertIn('decision_detail_checks',audit)
        text=checks_markdown(audit['decision_detail_checks'])
        self.assertIn('Outcome, amount and deadline',text)
        self.assertIn('INR 200000',text)
