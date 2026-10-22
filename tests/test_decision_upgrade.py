"""Adversarial checks for decision attribution and qualification retention."""
import unittest
from test_stage44 import document,decision
from stage44_brief import audit_brief
from stage44_attribution import attribution_map
from stage44_source import assemble

def outlined_pdf(text,final_x=30,method='native'):
    lines=[dict(text=t,x=final_x if t.startswith(('IV.','XIII.')) else 30 if t.startswith(('I.','II.','III.')) else 0,y=i*20,size=12)
           for i,t in enumerate(text.splitlines())]
    pages=[(lines,800)]
    diagnostics=[dict(page=1,method='ocr' if method=='mixed' else method,readable=True,replacement_characters=0,ocr_error=None)]
    if method=='mixed':
        pages.append(([],800))
        diagnostics.append(dict(page=2,method='native',readable=True,replacement_characters=0,ocr_error=None))
    return assemble('synthetic.pdf','fixture',pages,diagnostics,[])

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

    def test_merits_qualification_is_preserved_next_to_disposition(self):
        doc=document('1. It is made clear that this Court has not expressed any views on the merits of the matter.\n'
                     '2. In view of the above, the appeals fail and are hereby dismissed.')
        text,_=decision(doc)
        self.assertIn('not expressed any views on the merits',text)

    def test_party_merits_claim_does_not_become_current_qualification(self):
        doc=document('1. The respondent submits that this Court has not expressed any views on the merits.\n'
                     '2. The appeal is dismissed.')
        self.assertNotIn('respondent submits',decision(doc)[0])

    def test_ought_to_be_is_not_a_disposition(self):
        self.assertEqual(decision(document('1. The petition ought to be dismissed in limine.'))[0],'')

    def test_structural_final_heading_recovers_inline_quote_with_warning(self):
        doc=outlined_pdf('I. INTRODUCTION:\nThe claimant challenges the order.\nII. FACTS:\nThe dispute concerns payment.\n'
                     'III. ANALYSIS:\nThe dictionary defines the term as “an unresolved definition.\n'
                     'IV. CONCLUSION:\nThe appeal is allowed.\nThe respondent shall pay within six weeks.')
        text,result=decision(doc)
        self.assertIn('appeal is allowed',text)
        self.assertIn('QUOTE_ATTRIBUTION_REVIEW',audit_brief(doc,result)['flags'])

    def test_final_heading_alone_never_resets_a_quote(self):
        doc=outlined_pdf('The authority states “an unfinished quotation.\nIV. CONCLUSION:\nThe appeal is allowed.')
        self.assertEqual(decision(doc)[0],'')

    def test_long_outline_gap_still_recovers_with_attribution_warning(self):
        doc=outlined_pdf('I. INTRODUCTION:\nThe claim is disputed.\nII. FACTS:\nPayment was requested.\n'
                         'III. ANALYSIS:\nThe term means “an unfinished definition.\n'
                         'XIII. CONCLUSION:\nThe appeal is allowed.')
        text,result=decision(doc)
        self.assertIn('appeal is allowed',text)
        self.assertIn('QUOTE_ATTRIBUTION_REVIEW',audit_brief(doc,result)['flags'])

    def test_block_quotation_is_not_reset_by_matching_final_heading(self):
        doc=outlined_pdf('I. INTRODUCTION:\nThe claimant challenges the order.\nII. FACTS:\nThe dispute concerns payment.\n'
                     'III. ANALYSIS:\nThe earlier order reads:\n“1. This is quoted material.\n'
                     'IV. CONCLUSION:\nThe appeal is allowed.')
        self.assertEqual(decision(doc)[0],'')

    def test_indented_heading_cannot_recover_inline_quote(self):
        text='I. INTRODUCTION:\nThe claim is disputed.\nII. FACTS:\nPayment was requested.\nIII. ANALYSIS:\nThe term means “a definition.\nIV. CONCLUSION:\nThe appeal is allowed.'
        self.assertEqual(decision(outlined_pdf(text,final_x=90))[0],'')

    def test_plain_text_outline_does_not_claim_pdf_geometry(self):
        text='I. INTRODUCTION:\nThe claim is disputed.\nII. FACTS:\nPayment was requested.\nIII. ANALYSIS:\nThe term means “a definition.\nIV. CONCLUSION:\nThe appeal is allowed.'
        self.assertEqual(decision(document(text))[0],'')

    def test_ocr_outline_does_not_claim_native_font_geometry(self):
        text='I. INTRODUCTION:\nThe claim is disputed.\nII. FACTS:\nPayment was requested.\nIII. ANALYSIS:\nThe term means “a definition.\nIV. CONCLUSION:\nThe appeal is allowed.'
        for method in ['ocr','mixed']:
            with self.subTest(method=method):
                self.assertEqual(decision(outlined_pdf(text,method=method))[0],'')

if __name__=='__main__':unittest.main()
