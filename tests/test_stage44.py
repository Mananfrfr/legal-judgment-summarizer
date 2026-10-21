import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
from copy import deepcopy
import tempfile
import numpy as np
import fitz
from stage44_source import assemble,read_document,verify_evidence,sentence_ranges
from stage44_brief import build_brief,audit_brief,summarize_file
from stage44_windows import aggregate_windows,window_ranges

def document(*pages):
    lines=[([dict(text=t,x=0,y=i*20,size=12) for i,t in enumerate(page.splitlines()) if t.strip()],800) for page in pages]
    ds=[dict(page=i+1,method='text',characters=len(p),readable=True,replacement_characters=0,ocr_error=None) for i,p in enumerate(pages)]
    return assemble('synthetic.txt','test',lines,ds,[])

def decision(doc,budget=800):
    r=build_brief(doc,budget)
    return ' '.join(e['sentence'] for e in r['sections']['decision']),r

class EvidenceTests(unittest.TestCase):
    def test_cross_page_sentence_and_exact_offsets(self):
        d=document('1. We direct the insurer to pay Rs. 2,00,000 within','six weeks, subject to adjustment of payments already made.')
        text,r=decision(d)
        self.assertIn('six weeks',text)
        self.assertTrue(all(verify_evidence(e,d.pages) for e in d.sentences))
        self.assertEqual((r['sections']['decision'][0]['page'],r['sections']['decision'][0]['page_end']),(1,2))

    def test_amount_tampering_fails_closed(self):
        d=document('1. We direct payment of Rs. 20,000 within six weeks.')
        _,r=decision(d)
        r['sections']['decision'][0]['sentence']=r['sections']['decision'][0]['sentence'].replace('20,000','200,000')
        self.assertEqual(audit_brief(d,r)['status'],'blocked_source_mismatch')

    def test_negation_tampering_detected(self):
        d=document('1. We clarify that this order does not decide the merits.')
        _,r=decision(d)
        r['sections']['decision'][0]['sentence']=r['sections']['decision'][0]['sentence'].replace('not ','')
        self.assertFalse(audit_brief(d,r)['exact_source_check_passed'])

    def test_wrong_page_detected(self):
        d=document('1. We dismiss this appeal for the reasons above.','2. There is no order as to costs.')
        e=deepcopy(d.sentences[0]);e['page']=2
        self.assertFalse(verify_evidence(e,d.pages))

    def test_legal_abbreviations_and_decimals(self):
        text='M/s Acme Co. Ltd. shall deposit Rs. 20.50 at P.S. G.D. Road. The appeal is allowed.'
        values=[text[a:b] for a,b in sentence_ranges(text)]
        self.assertEqual(len(values),2);self.assertIn('P.S. G.D. Road.',values[0])

    def test_quoted_earlier_court_not_current_order(self):
        d=document('1. The earlier court held as follows:\n“2. We direct payment of Rs. 500 within two weeks.”\n3. We dismiss this appeal. No order as to costs.')
        text,r=decision(d)
        self.assertNotIn('Rs. 500',text);self.assertIn('dismiss this appeal',text)

    def test_inline_historical_quote_excluded(self):
        d=document('1. The earlier judgment said: “We allow this appeal. The order is set aside.”\n2. We dismiss the present appeal.')
        text,_=decision(d)
        self.assertNotIn('We allow this appeal',text);self.assertIn('We dismiss',text)

    def test_referral_without_final_merits_outcome(self):
        d=document('1. Whether the statutory provision applies is the issue before us.\n2. We direct the Registry to place this matter before a larger Bench.')
        text,_=decision(d)
        self.assertIn('larger Bench',text);self.assertNotIn('dismissed',text)

    def test_unusual_disposition_and_all_conditions(self):
        d=document('1. The appeals succeed.\n2. We direct the debtor to deposit Rs. 100 within eight weeks.\n3. The stay shall continue subject to that deposit.\n4. The Registry shall renew the fixed deposit every six months.')
        text,_=decision(d)
        for value in ['succeed','eight weeks','subject to','six months']: self.assertIn(value,text)

    def test_long_modified_subject_dismissal(self):
        d=document('1. The Arbitration Petition seeking appointment of an independent arbitrator stands dismissed.')
        self.assertIn('stands dismissed',decision(d)[0])

    def test_record_of_proceedings_not_repeated(self):
        d=document('1. We allow this appeal.','RECORD OF PROCEEDINGS\n1. The appeal is dismissed.')
        text,_=decision(d)
        self.assertNotIn('dismissed',text);self.assertIn('allow',text)

    def test_small_budget_never_cuts_order(self):
        d=document('1. We direct payment within six weeks subject to '+'the original terms and conditions '*30+'.')
        _,r=decision(d,100)
        self.assertIn('BUDGET_OVERFLOW',audit_brief(d,r)['flags'])
        self.assertTrue(r['sections']['decision'][0]['sentence'].endswith('.'))

    def test_missing_outcome_abstains(self):
        d=document('1. The claimant seeks payment from the company.\n2. Counsel submitted that the appeal should be allowed.')
        text,r=decision(d)
        self.assertFalse(text);self.assertIn('decision',audit_brief(d,r)['missing_sections'])

    def test_independent_audit_catches_omitted_second_deadline(self):
        d=document('1. We direct payment of Rs. 100 within two weeks.\n2. The debtor shall deposit Rs. 200 within four weeks.')
        _,r=decision(d)
        r['sections']['decision']=[e for e in r['sections']['decision'] if '200' not in e['sentence']]
        audit=audit_brief(d,r)
        self.assertTrue(any('200' in e['sentence'] for e in audit['possible_omitted_directions']))

    def test_numeric_date_not_paragraph_number(self):
        d=document('1. The claim was submitted on\n31.01.2007. We dismiss the appeal.')
        self.assertIn('31.01.2007',d.paragraphs[0]['text'])

    def test_inline_word_order_not_heading(self):
        d=document('1. We set aside the impugned order granting unconditional stay.')
        self.assertIn('impugned order granting',d.sentences[0]['sentence'])

    def test_poor_page_is_flagged(self):
        d=document('1. We dismiss this appeal.')
        d.diagnostics.append(dict(page=2,method='native',readable=False,ocr_error='timeout',replacement_characters=0))
        _,r=decision(d)
        self.assertIn('EXTRACTION_REVIEW',audit_brief(d,r)['flags'])

    def test_no_claim_of_legal_accuracy(self):
        d=document('1. We allow the appeal.')
        _,r=decision(d);a=audit_brief(d,r)
        self.assertTrue(a['human_validation_required']);self.assertFalse(a['legal_accuracy_measured'])

    def test_issue_intro_can_mention_counsel_without_being_an_argument(self):
        d=document('1. Having heard learned counsel, the only question for consideration is whether the limitation rule applies.\n2. We dismiss this appeal.')
        r=build_brief(d)
        self.assertIn('only question',' '.join(e['sentence'] for e in r['sections']['issue']))

    def test_current_procedural_context_beats_historical_model_label(self):
        d=document('1. The present two sets of appeals arise from the impugned promotion order.\n2. The applicant seeks appointment as an engineer.\n3. We dismiss these appeals.')
        r=build_brief(d,predictions={1:dict(role_code='ISSUE',role_confidence=.99,importance_score=.99)})
        issue=' '.join(e['sentence'] for e in r['sections']['issue'])
        self.assertIn('present two sets',issue);self.assertNotIn('appointment as an engineer',issue)

    def test_batch_failure_not_reported_as_success(self):
        from stage44_cli import run_batch
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);p=folder/'valid.txt'
            p.write_text('1. The only issue is whether the provision applies.\n2. We allow this appeal.',encoding='utf8')
            report,archive=run_batch([p,folder/'missing.pdf'],folder/'out',ocr='off')
            self.assertEqual((report['processed'],report['failed']),(1,1))
            self.assertEqual(report['machine_source_checks_passed'],0)
            self.assertTrue(Path(archive).exists())

    def test_ancillary_permission_not_mistaken_for_final_outcome(self):
        d=document('1. Interlocutory Application No. 3 is allowed granting permission to produce additional documents.\n2. The court has heard the arguments on the main appeal.')
        self.assertFalse(decision(d)[0])

    def test_reported_appellate_ruling_is_not_current_order(self):
        d=document('1. The first appellate court went further to find that the suit has to be dismissed.\n2. We allow the present appeal and set aside that judgment.')
        text,_=decision(d)
        self.assertIn('We allow',text);self.assertNotIn('has to be dismissed',text)

    def test_superscript_marker_does_not_change_section_number(self):
        # Synthetic geometry avoids relying on case-specific names or numbers.
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'footnote.pdf'
            pdf=fitz.open();page=pdf.new_page()
            page.insert_text((72,80),'1. The issue concerns Section 125',fontsize=14)
            end=72+fitz.get_text_length('1. The issue concerns Section 125',fontsize=14)
            page.insert_text((end,75),'6',fontsize=8)
            page.insert_text((end+5,80),' of the Act.',fontsize=14)
            page.insert_text((72,110),'2. We dismiss this appeal.',fontsize=14)
            pdf.save(p);pdf.close()
            d=read_document(p,ocr='off')
            self.assertNotIn('1256',' '.join(s['sentence'] for s in d.sentences))
            self.assertTrue(all(verify_evidence(e,d.pages) for e in d.sentences))

    def test_blank_pdf_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'blank.pdf';pdf=fitz.open();pdf.new_page();pdf.save(p);pdf.close()
            with self.assertRaises(ValueError):read_document(p,ocr='off')

class WindowTests(unittest.TestCase):
    def test_long_document_covers_every_sentence(self):
        for n in [1,119,120,121,240,537,2001]:
            values,counts=aggregate_windows(n,lambda a,b:np.arange(a,b)[:,None])
            np.testing.assert_allclose(values[:,0],np.arange(n))
            self.assertTrue((counts>0).all());self.assertLessEqual(max(b-a for a,b in window_ranges(n)),120)

    def test_missing_or_nonfinite_model_values_fail(self):
        with self.assertRaises(ValueError):aggregate_windows(10,lambda a,b:np.zeros((b-a-1,2)))
        with self.assertRaises(ValueError):aggregate_windows(10,lambda a,b:np.full((b-a,2),np.nan))

    def test_overlap_averages_instead_of_duplicate_evidence(self):
        values,counts=aggregate_windows(150,lambda a,b:np.ones((b-a,1))*a)
        self.assertEqual(counts[100],2);self.assertEqual(values[100,0],45)

class LayoutRecoveryTests(unittest.TestCase):
    def test_unclosed_indented_quote_recovers_at_body_paragraph(self):
        lines=[dict(text='1. The provision is reproduced below:',x=70,y=30,size=12),
               dict(text='“138. A person shall pay the statutory penalty.',x=110,y=60,size=12),
               dict(text='2. We find that the notice was served correctly.',x=70,y=90,size=12),
               dict(text='3. The appeal is allowed and the conviction stands restored.',x=70,y=120,size=12)]
        d=assemble('layout.pdf','hash',[(lines,800)],[],[])
        self.assertTrue(d.sentences[1]['quoted'])
        self.assertFalse(d.sentences[-1]['quoted'])
        self.assertTrue(any('boundary recovered' in w for w in d.warnings))
        self.assertIn('conviction stands restored',decision(d)[0])
        self.assertTrue(all(verify_evidence(s,d.pages) for s in d.sentences))

    def test_quoted_order_does_not_escape_on_same_indent_numbering(self):
        lines=[dict(text='1. The earlier court said:',x=70,y=30,size=12),
               dict(text='“10. The statutory exception applies.',x=110,y=60,size=12),
               dict(text='11. We allow this appeal and direct payment.',x=110,y=90,size=12)]
        d=assemble('layout.pdf','hash',[(lines,800)],[],[])
        self.assertFalse(decision(d)[0])
        self.assertTrue(d.sentences[-1]['quoted'])

    def test_without_geometry_unclosed_quote_stays_flagged(self):
        d=document('1. The earlier court said:\n“10. The exception applies.\n11. We allow this appeal.')
        self.assertFalse(decision(d)[0])
        self.assertTrue(any('Unbalanced quotation' in w for w in d.warnings))

    def test_mixed_double_quote_style_can_close(self):
        d=document('1. The earlier court said:\n“10. We allow this appeal."\n2. We dismiss the present appeal.')
        self.assertNotIn('We allow',decision(d)[0]);self.assertIn('We dismiss',decision(d)[0])

    def test_criminal_revision_abbreviation_keeps_context(self):
        d=document('1. The appeal challenges an order in Crl. R.P. No. 865 of 2023. We allow the appeal.')
        self.assertEqual(len(d.sentences),2)
        self.assertIn('Crl. R.P. No. 865',d.sentences[0]['sentence'])

    def test_precedent_question_is_not_current_issue(self):
        d=document('1. In Alpha v. Beta, the issue before this Court was whether notice was served.\n2. The present appeal arises from refusal to award compensation.\n3. We dismiss this appeal.')
        r=build_brief(d)
        self.assertNotIn('Alpha',' '.join(s['sentence'] for s in r['sections']['issue']))

    def test_third_person_current_assessment_recognized(self):
        d=document('1. This Court is of the considered view that the statutory notice was correctly served. The burden was not discharged.\n2. The appeal is allowed.')
        r=build_brief(d)
        self.assertIn('burden',' '.join(s['sentence'] for s in r['sections']['reasoning']))

    def test_submission_heading_context_ends_at_reasoning_heading(self):
        d=document('ARGUMENTS ON BEHALF OF THE APPELLANT\n1. He argued that the claim was barred. Therefore the petition should fail.\nREASONING\n2. This Court is of the view that the claim was made within time.\n3. We allow the petition.')
        r=build_brief(d)
        reasons=' '.join(s['sentence'] for s in r['sections']['reasoning'])
        self.assertNotIn('should fail',reasons);self.assertIn('within time',reasons)


class ReviewRegressionTests(unittest.TestCase):
    def test_cr_pc_variants_preserve_sentence_and_source(self):
        for abbreviation in ['Cr. PC', 'Cr. P.C.', 'Cr.P.C.']:
            d=document(f'1. We invoke Section 482, {abbreviation} and quash the FIR. The appeal stands allowed.')
            self.assertEqual(len(d.sentences),2)
            self.assertIn('quash the FIR',d.sentences[0]['sentence'])
            self.assertTrue(all(verify_evidence(s,d.pages) for s in d.sentences))

    def test_letter_with_first_person_is_not_court_reasoning(self):
        d=document('1. The following letter was written by the company: “Dear Sirs, the currency depreciation has added to our woes. We cannot continue at this price. Yours faithfully, the manager.”\n2. We find that the contractual obligation remains binding. No evidence of fraud has been produced.\n3. We dismiss this appeal.')
        r=build_brief(d,predictions={s['id']:dict(role_code='ANALYSIS',role_confidence=.99,importance_score=.99) for s in d.sentences})
        reasons=' '.join(s['sentence'] for s in r['sections']['reasoning'])
        self.assertIn('contractual obligation',reasons)
        self.assertNotIn('our woes',reasons);self.assertNotIn('at this price',reasons)

    def test_party_paragraph_cannot_be_promoted_by_model_score(self):
        d=document('1. Counsel submitted that the agreement was void. Therefore, the award could not be enforced. The claim was barred by limitation.\n2. In our view, those objections have no evidentiary basis. The agreement was performed voluntarily.\n3. We dismiss this appeal.')
        r=build_brief(d,predictions={s['id']:dict(role_code='RATIO',role_confidence=1,importance_score=1) for s in d.sentences})
        reasons=' '.join(s['sentence'] for s in r['sections']['reasoning'])
        self.assertNotIn('barred by limitation',reasons)
        self.assertIn('performed voluntarily',reasons)

    def test_short_quoted_we_does_not_establish_court_voice(self):
        d=document('1. The complaint relies on the phrase “we are not denying our obligation” in the letter.\n2. We find the complaint unsupported by the records.\n3. We dismiss this appeal.')
        r=build_brief(d)
        self.assertFalse(any('phrase' in s['sentence'] for s in r['sections']['reasoning']))

    def test_reasoning_keeps_consecutive_context_beyond_five_sentences(self):
        d=document('1. The issue before us is whether the agreement is binding.\n2. We find that the agreement remains binding. The parties signed it voluntarily. The price was settled in advance. Both sides performed the first delivery. There is no evidence of deception. The subsequent price change does not invalidate consent. Therefore the objection has no factual basis.\n3. We dismiss this appeal.')
        r=build_brief(d)
        reasons=r['sections']['reasoning']
        self.assertEqual(len(reasons),7)
        self.assertEqual(len({s['reasoning_block_id'] for s in reasons}),1)
        self.assertEqual([s['id'] for s in reasons],list(range(reasons[0]['id'],reasons[-1]['id']+1)))

    def test_hearing_and_counsel_list_not_reasoning(self):
        d=document('1. We have heard Mr. Smith for the appellant and Ms. Jones for the respondent. We have carefully considered the submissions and perused the records.\n2. We find that the claim is barred by limitation.\n3. We dismiss this appeal.')
        r=build_brief(d)
        self.assertFalse(any('heard' in s['sentence'] or 'perused' in s['sentence'] for s in r['sections']['reasoning']))

    def test_referral_analysis_is_separate_from_direction(self):
        d=document('1. The question before us is whether the statutory exception applies.\n2. We are of the opinion that the existing interpretation is inconsistent. We cannot agree with the earlier view. The provision overrides the general rule for compensation arising from injury. We direct the Registry to place the matter before a larger Bench.')
        text,r=decision(d)
        self.assertIn('larger Bench',text);self.assertNotIn('inconsistent',text)
        self.assertIn('inconsistent',' '.join(s['sentence'] for s in r['sections']['reasoning']))

    def test_split_prefix_never_removes_quashing_or_condition(self):
        d=document('1. We are of the opinion that the High Court erred. The High Court having failed to exercise its power, we invoke it and quash the FIR on which no further proceedings will be taken. The appeal stands allowed.')
        text,_=decision(d)
        self.assertIn('quash the FIR',text);self.assertIn('no further proceedings',text)

    def test_competing_positions_dont_mix_in_court_conclusion(self):
        d=document('1. The petitioner seeks appointment of an arbitrator. The respondent contends that the claim is barred by limitation.\n2. Even if limitation is computed from that date, the claim stands barred when the notice was issued seeking appointment of an arbitrator.\n3. The petition stands dismissed.')
        r=build_brief(d)
        self.assertEqual(r['issue_mode'],'reported_competing_positions')
        issue=' '.join(s['sentence'] for s in r['sections']['issue'])
        self.assertIn('respondent contends',issue);self.assertNotIn('Even if',issue)

    def test_legal_dispute_description_beats_filing_history(self):
        d=document('1. The present appeal arises from an order dated 01.01.2020.\n2. The policy was challenged on the ground that the claimant had been placed below another employee in seniority, and from that point the present lis has arisen.\n3. We dismiss this appeal.')
        r=build_brief(d)
        self.assertEqual(r['issue_mode'],'source_dispute_description')
        self.assertIn('seniority',r['sections']['issue'][0]['sentence'])

    def test_reported_deadline_retained_in_audit_without_current_warning(self):
        d=document('1. Counsel submitted that the debtor shall deposit money within four weeks.\n2. We dismiss the present appeal.')
        r=build_brief(d);a=audit_brief(d,r)
        self.assertFalse(a['possible_omitted_directions'])
        self.assertEqual(len(a['reported_or_historical_direction_mentions']),1)

    def test_roman_list_marker_is_never_an_isolated_sentence(self):
        text='The following objections were raised: i. Supply was impossible. ii. There was no loss. The appeal stands dismissed.'
        values=[text[a:b] for a,b in sentence_ranges(text)]
        self.assertIn('ii. There was no loss.',values)
        self.assertNotIn('ii.',values)

    def test_elapsed_time_in_history_is_not_a_deadline(self):
        d=document('1. Disputes arose within twelve days from the signing of the agreement.\n2. We dismiss this appeal.')
        r=build_brief(d);a=audit_brief(d,r)
        self.assertFalse(a['possible_omitted_directions'])
        self.assertEqual(len(a['reported_or_historical_direction_mentions']),1)

    def test_heading_fragment_not_selected_as_reason(self):
        d=document('1. We find that the objection lacks evidence. LEGAL FRAMEWORK TO DETERMINE LIABILITY\n2. We dismiss this appeal.')
        r=build_brief(d)
        self.assertNotIn('LEGAL FRAMEWORK',' '.join(s['sentence'] for s in r['sections']['reasoning']))

    def test_unaccompanied_quote_introduction_not_selected_as_reason(self):
        d=document('1. Here, we would like to reproduce the earlier order, which is as follows:\n“2. The claimant shall pay the costs.”\n3. We find the current claim unsupported by evidence.\n4. We dismiss this appeal.')
        r=build_brief(d)
        self.assertNotIn('reproduce',' '.join(s['sentence'] for s in r['sections']['reasoning']))

    def test_explicit_question_without_whether(self):
        d=document('1. This brings us to the next question, from which date will maintenance be payable?\n2. In our view it must run from the application date.\n3. We allow this appeal.')
        r=build_brief(d)
        self.assertEqual(r['issue_mode'],'explicit_source_issue')
        self.assertIn('from which date',r['sections']['issue'][0]['sentence'])

if __name__=='__main__':unittest.main()
