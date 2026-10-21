"""Conservative, extractive case brief. Model scores assist ranking, never create text."""
from __future__ import annotations
from copy import deepcopy
import re
from stage44_source import read_document, verify_evidence
from stage44_attribution import attribution_map, current_analysis, COURT, ASSESSMENT, PARTY, LEGAL_DISPUTE, assessment, PASSIVE_REPORTED

VERSION = '4.6.0-preview'
SECTION_TITLES = {'background':'Background / allegations and procedural history',
                  'issue':'Issues / dispute context', 'reasoning':'Reasoning (consecutive source passages; attribution needs review)',
                  'decision':'Decision and operative directions'}
HISTORICAL = re.compile(r'\b(?:had\s+(?:been\s+)?(?:dismissed|allowed|directed)|(?:High|Trial|Family|District|lower|appellate)\s+Court\s+(?:had|held|found|concluded|went|observed|ordered|directed|dismissed|allowed)|(?:counsel|appellant|respondent|petitioner)\s+(?:submitted|argued|contended|urged|prayed)|it\s+was\s+(?:held|ordered|directed|submitted)|in\s+the\s+case\s+of)\b',re.I)
SUBJECT_OUTCOME = re.compile(r'\b(?:appeals?|petitions?|applications?|suit|(?:present|this|these)\s+(?:batch|group)\s+of\s+(?:matters|cases))\b.{0,110}?\b(?:stands?\s+(?:dismissed|allowed|disposed)|(?:is|are|be)\s+(?:hereby\s+|accordingly\s+|partly\s+|partially\s+)?(?:dismissed|allowed|disposed|rejected)|succeeds?|fails?)\b',re.I)
DIRECT = re.compile(r'\b(?:we\s+(?:(?:hereby|accordingly|therefore|thus|further|also)\s+)*(?:direct|allow|dismiss|quash|set\s+aside|remand|restore|grant|clarify|modify|order|dispose|permit)|in\s+allowing\s+(?:the|this)\s+appeal)\b',re.I)
ACTION = re.compile(r'\b(?:shall|is\s+directed|are\s+directed|set\s+aside|restored|quash(?:ed)?|modified|remand(?:ed)?|remit(?:ted)?|deposit(?:ed)?|payable|enhanced|release(?:d)?\s+on\s+bail|subject\s+to|exemption\b.{0,40}\bgranted|no\s+(?:order\s+as\s+to\s+)?costs|within\b.{0,50}\b(?:days?|weeks?|months?))\b',re.I)
ISSUE = re.compile(r'\b(?:whether|questions?\s+(?:of\s+law|for\s+consideration|that\s+arises?|arising)|(?:main|sole|primary|principal|central|singular|next)\s+(?:issue|question|grievance)|issue\s+(?:before\s+us|concerning)|we\s+are\s+called\s+upon|controversy\s+(?:is|relates))\b',re.I)
CONTEXT = re.compile(r'\b(?:seeks?|seeking|challeng\w*|aggrieved|assail\w*|prayer|grievance|contends?|disputes?|appeal\s+(?:arises|is\s+directed)|rejection\s+of|rejected\s+.*bail|appointment\s+of\s+.*arbitrator)\b',re.I)
PROCEDURAL=re.compile(r'\b(?:(?:present|instant|these|this)\b.{0,30}\b(?:appeals?|petitions?)\b.{0,45}\b(?:arise|arises|against|challeng)|(?:appeals?|petitions?)\b.{0,45}\b(?:arise|arises)\s+(?:from|out\s+of)|challeng\w*\b.{0,30}\bpresent\b)',re.I)
REASON = re.compile(r'\b(?:we\s+(?:find|hold|are\s+of|have|cannot|do\s+not|agree)|in\s+our\s+(?:view|opinion)|therefore|because|it\s+(?:follows|is\s+(?:clear|evident))|prima\s+facie|goes\s+without\s+saying|barred\s+by\s+limitation|no\s+(?:merit|substance))\b',re.I)
FINAL_CONNECTOR = re.compile(r'\b(?:accordingly|in\s+view\s+of|for\s+(?:the\s+)?(?:above|these|foregoing)|in\s+the\s+result|consequently|hence)\b',re.I)
ORDER_DETAIL = re.compile(r'\b(?:shall|within|subject\s+to|entitled\s+to|stands?\s+restored|set\s+aside|quash(?:ed)?|no\s+further\s+proceedings|enhanced|payable|Rs\.|rupees|compensation\s+(?:is|of)|maintenance\s+(?:is|of|at)|released\s+on\s+bail|no\s+order\s+as\s+to\s+costs|will\s+have\s+to\s+be\s+placed)\b|₹|\d\s*%',re.I)

def elapsed_history(text):
    return bool(re.search(r'\bwithin\s+(?:a\s+period\s+of\s+)?(?:\d+|[a-z]+)\s+(?:days?|weeks?|months?)\s+(?:from|of|after)\s+(?:the\s+)?(?:signing|execution|filing|occurrence|accident)\b',text,re.I)) and not re.search(r'\b(?:shall|must|direct|directed|order|ordered)\b',text,re.I)

def outcome_text(text):
    """Ignore optional commas for cue matching only; evidence remains verbatim."""
    return re.sub(r'\s+', ' ', text.replace(',', ' '))

def outcome_score(s, total):
    text=outcome_text(s['sentence'])
    if s['quoted'] or text.rstrip().endswith('?'):
        return 0
    direct=bool(DIRECT.search(text))
    disposition=bool(SUBJECT_OUTCOME.search(text))
    if re.search(r'\b(?:interlocutory\s+application|permission\s+to\s+(?:produce|file)|exemption\s+from)\b',text,re.I):
        return 0 # Ancillary procedural relief cannot establish the main case outcome.
    if not (direct or disposition):
        return 0
    # Requests and reports of earlier orders cannot alone anchor a present order.
    if (HISTORICAL.search(text) or PASSIVE_REPORTED.search(text)) and not direct:
        return 0
    if re.search(r'\b(?:prays?|prayed|seeks?|seeking|request(?:ed)?)\b.{0,45}\b(?:be\s+allowed|be\s+dismissed)',text,re.I):
        return 0
    if re.search(r'\b(?:not|never)\s+(?:hereby\s+)?(?:allow|dismiss|quash)',text,re.I):
        return 0
    score=4*direct+3*disposition+2*bool(FINAL_CONNECTOR.search(text))+s['id']/max(1,total-1)
    if re.search(r'^Pending\s+applications',text,re.I):
        score-=3
    return score

def _entry(s, reason):
    e=deepcopy(s)
    e['selection_reason']=reason
    return e

def build_brief(doc, budget=800, predictions=None, model_info=None):
    if not 100 <= int(budget) <= 3000:
        raise ValueError('Choose a budget between 100 and 3000 words.')
    predictions=predictions or {}
    sections={k:[] for k in SECTION_TITLES}
    warnings=list(doc.warnings)
    sentences=doc.sentences
    attributes=attribution_map(doc)
    eligible=[s for s in sentences if not s['quoted'] and len(s['sentence'].split()) >= 3]
    scores={s['id']:outcome_score(s,len(sentences)) for s in eligible}
    excluded_roles={'party_submission','quoted_material','document_or_correspondence','reported_material'}
    anchors=[s for s in eligible if scores[s['id']] >= 3 and attributes[s['id']]['label'] not in excluded_roles]
    decision_ids=set()
    moved_analysis_ids=set()
    if anchors:
        # Strongest order anchor, with a finality tie-break; no page-fraction cutoff.
        anchor=max(anchors,key=lambda s:(scores[s['id']],s['id']))
        ap=anchor['paragraph_id']
        cluster={ap}
        # Preserve nearby order paragraphs, including conditions without 'allowed'.
        for s in anchors:
            if abs(s['paragraph_id']-ap)<=8:
                cluster.add(s['paragraph_id'])
        left,right=min(cluster),max(cluster)
        for p in doc.paragraphs:
            if left-2 <= p['id'] <= right+8 and not p['quoted']:
                if (ACTION.search(p['text']) or SUBJECT_OUTCOME.search(outcome_text(p['text']))) and not elapsed_history(p['text']) and not p['text'].rstrip().endswith('?'):
                    # Do not extend past a postscript into historical narration.
                    if not (HISTORICAL.search(p['text']) or PASSIVE_REPORTED.search(p['text'])) or DIRECT.search(outcome_text(p['text'])):
                        cluster.add(p['id'])
        for s in eligible:
            if s['paragraph_id'] in cluster:
                decision_ids.add(s['id'])
                sections['decision'].append(_entry(s,'operative paragraph; preserve conditions and context'))
        # Mixed conclusion paragraphs can first explain the law, then give an
        # order. Move a purely analytical prefix into the reasoning pool; never
        # cut an amount, qualification, deadline or preceding operative clause.
        for pid in cluster:
            members=[s for s in sections['decision'] if s['paragraph_id']==pid]
            first_order=next((i for i,s in enumerate(members) if DIRECT.search(outcome_text(s['sentence'])) or SUBJECT_OUTCOME.search(outcome_text(s['sentence'])) or re.search(r'will\s+have\s+to\s+be\s+placed',s['sentence'],re.I)),0)
            prefix=members[:first_order]
            if prefix and all(current_analysis(s,attributes) and not ORDER_DETAIL.search(s['sentence']) for s in prefix):
                move={s['id'] for s in prefix}
                sections['decision']=[s for s in sections['decision'] if s['id'] not in move]
                decision_ids-=move
                moved_analysis_ids|=move
    else:
        warnings.append('No reliable present-court disposition located. Do not infer that an appeal was allowed or dismissed.')

    def pred(s):
        return predictions.get(s['id'],{})
    def role(s,codes):
        p=pred(s)
        return float(p.get('role_confidence',0)) if p.get('role_code') in codes else 0
    def importance(s):
        return float(pred(s).get('importance_score',0))
    pool=[s for s in eligible if s['id'] not in decision_ids]
    strong_issue=re.compile(r'\b(?:(?:only|main|sole|principal|central|singular)\s+(?:issue|question)|question\b.{0,50}\b(?:arises|consideration)|issue\s+before\s+us|primary\s+grievance)\b',re.I)
    issue_pool=[s for s in pool if ISSUE.search(s['sentence']) and attributes[s['id']]['label'] not in excluded_roles and not re.match(r'^(?:learned\s+(?:senior\s+)?counsel|it\s+was\s+submitted)\b',s['sentence'],re.I)]
    if any(strong_issue.search(s['sentence']) for s in issue_pool):
        issue_pool=[s for s in issue_pool if strong_issue.search(s['sentence']) or (re.match(r'^Whether\b',s['sentence']) and '?' in s['sentence'])]
    issue_label='explicit issue wording in source; attribution still needs review'
    issue_mode='explicit_source_issue'
    if not issue_pool:
        # Prefer an actual description of what is contested over filing history.
        dispute=[s for s in pool if LEGAL_DISPUTE.search(s['sentence']) and attributes[s['id']]['label'] not in excluded_roles]
        competing=[]
        for p in doc.paragraphs:
            ms=[s for s in pool if s['paragraph_id']==p['id']]
            if p['id'] <= max(2,len(doc.paragraphs)*.15) and re.search(r'\b(?:seeks|seeking)\b',p['text'],re.I) and PARTY.search(p['text']):
                # Keep both sides of a short, introductory dispute description.
                # This is labeled as reported positions, never court findings.
                competing=[s for s in ms if not ASSESSMENT.search(s['sentence']) or attributes[s['id']]['label']=='party_submission']
                if competing: break
        if dispute:
            issue_pool=dispute;issue_mode='source_dispute_description'
            issue_label='source description of the legal dispute; no reformulated question'
        elif competing:
            issue_pool=competing;issue_mode='reported_competing_positions'
            issue_label='reported competing positions; not findings or an explicit question'
        else:
            issue_pool=[s for s in pool if (CONTEXT.search(s['sentence']) or PROCEDURAL.search(s['sentence'])) and not ASSESSMENT.search(s['sentence']) and attributes[s['id']]['label'] not in excluded_roles]
            procedural=[s for s in issue_pool if PROCEDURAL.search(s['sentence'])]
            if procedural: issue_pool=procedural
            issue_mode='procedural_context'
            issue_label='procedural dispute context; not an explicit question formulated by the court'
        warnings.append('No explicit issue formulation selected: the issue section contains dispute context only.')
    issue_pool.sort(key=lambda s: (3*bool(ISSUE.search(s['sentence']))+2*role(s,{'ISSUE'})+3*bool(strong_issue.search(s['sentence']))+5*bool(PROCEDURAL.search(s['sentence']))+
                                  bool(re.search(r'\b(?:before\s+us|present|this\s+(?:appeal|petition))\b',s['sentence'],re.I))+
                                  2*bool(re.search(r'\b(?:present\s+appeal|appeal\s+arises|appeal\s+is\s+directed|aggrieved|refusal|appointment|contends)\b',s['sentence'],re.I))+
                                  importance(s)-1.5*bool(HISTORICAL.search(s['sentence'])), -s['id']), reverse=True)
    used=set(decision_ids)
    # Reserve space for background and reasons; whole sentences only.
    word_count=lambda es:sum(len(e['sentence'].split()) for e in es)
    decision_words=word_count(sections['decision'])
    remaining=max(0,int(budget)-decision_words)
    limits={'issue':max(80,int(remaining*.20)), 'background':max(80,int(remaining*.20)),
            'reasoning':max(100,int(remaining*.60))}

    def add_ranked(section, ranked, max_items, why):
        for s in ranked:
            if s['id'] in used:
                continue
            if len(sections[section])>=max_items:
                break
            # If a crucial single sentence is longer than its share, preserve it and flag overflow.
            if sections[section] and word_count(sections[section])+len(s['sentence'].split())>limits[section]:
                continue
            sections[section].append(_entry(s,why)); used.add(s['id'])

    add_ranked('issue',issue_pool,2 if issue_mode in {'explicit_source_issue','reported_competing_positions'} else 1,issue_label)
    background=sorted(pool,key=lambda s:(2*role(s,{'FAC','PREAMBLE'})+importance(s)+
                       2*(1-s['id']/max(1,len(sentences)))+bool(CONTEXT.search(s['sentence'])), -s['id']),reverse=True)
    # Reserve court-analysis passages before background, so ranking cannot use
    # up a key reasoning sentence in the background section.
    background=[s for s in background if not current_analysis(s,attributes)]
    add_ranked('background',background,3,'background / reported allegations; not a finding of truth')
    limits['reasoning']=max(limits['reasoning'],int(budget)-sum(word_count(v) for v in sections.values()))
    reasoning_blocks=[]
    for p in doc.paragraphs:
        run=[]
        def flush_run():
            if not run:return
            # Consecutive sentences form the unit of selection. Long paragraphs
            # are split at sentence boundaries and explicitly marked as excerpts.
            candidates=[];chunk=[];words=0
            for s in run:
                n=len(s['sentence'].split())
                introduces_list=chunk and bool(re.search(r'\b(?:arguments|grounds|reasons|following)\b.*:',chunk[-1]['sentence'],re.I))
                if chunk and words+n>max(160,int(limits['reasoning']*.55)) and not introduces_list:
                    candidates.append(chunk);chunk=[];words=0
                chunk.append(s);words+=n
            if chunk:candidates.append(chunk)
            for chunk in candidates:
                if not any(assessment(s['sentence']) for s in chunk):continue
                text=' '.join(s['sentence'] for s in chunk)
                score=max(importance(s)+role(s,{'ANALYSIS','RATIO'}) for s in chunk)
                score+=2*sum(assessment(s['sentence']) for s in chunk)/len(chunk)
                score+=chunk[-1]['id']/max(1,len(sentences))
                if any(s['id'] in moved_analysis_ids for s in chunk):score+=1
                if re.search(r'\b(?:postscript|before\s+we\s+part)\b',p['text'],re.I):score-=3
                reasoning_blocks.append(dict(entries=list(chunk),score=score,tokens=set(re.findall(r'\b[a-z]{4,}\b',text.lower()))))
        for s in [s for s in sentences if s['paragraph_id']==p['id']]:
            if s['id'] not in used and current_analysis(s,attributes) and not s['quoted']:
                run.append(s)
            else:
                flush_run();run=[]
        flush_run()
    selected_tokens=set()
    while reasoning_blocks:
        def block_score(b):
            overlap=len(b['tokens'] & selected_tokens)/max(1,len(b['tokens'] | selected_tokens))
            return b['score']-2*overlap
        block=max(reasoning_blocks,key=block_score);reasoning_blocks.remove(block)
        words=word_count(block['entries'])
        if sections['reasoning'] and word_count(sections['reasoning'])+words>limits['reasoning']:continue
        for s in block['entries']:
            entry=_entry(s,'consecutive passage with current-court assessment cues; attribution is heuristic')
            entry['reasoning_block_id']=block['entries'][0]['id']
            sections['reasoning'].append(entry);used.add(s['id'])
        selected_tokens|=block['tokens']
    for entries in sections.values():
        entries.sort(key=lambda s:s['id'])
        for entry in entries:
            entry['attribution']=attributes[entry['id']]
            if entry['id'] in predictions:
                entry['ranking_metadata']=dict(predictions[entry['id']])
                entry['ranking_metadata']['calibrated_probability']=False
    count=sum(word_count(v) for v in sections.values())
    if count>budget:
        warnings.append(f'Budget exceeded ({count}/{budget} words) to keep complete evidence and operative conditions. No sentence was truncated.')
    if not predictions:
        warnings.append('Evidence-only mode: trained encoder and selector were not used.')
    if model_info and model_info.get('long_sentence_ids'):
        warnings.append(f'{len(model_info["long_sentence_ids"])} sentences exceed the trained encoder target length; model ranking sees their first 192 tokens. Full text is retained for evidence checks.')
    return dict(version=VERSION, source_file=doc.source_file,source_sha256=doc.sha256,
                caption_verbatim=doc.caption,sections=sections,issue_mode=issue_mode,word_limit=int(budget),summary_words=count,
                warnings=list(dict.fromkeys(warnings)), pdf_diagnostics=doc.diagnostics,
                canonical_pages=doc.pages,model=model_info or {'mode':'evidence_only'},
                total_source_sentences=len(sentences),quoted_source_sentences=sum(s['quoted'] for s in sentences))

def audit_brief(doc, result):
    """Independent evidence validation plus broad omission screening.

    A machine-check success is NOT a claim of legal correctness or complete recall.
    This screening deliberately does not call the summary's outcome scorer.
    """
    selected=[e for es in result['sections'].values() for e in es]
    invalid=[e['id'] for e in selected if not verify_evidence(e,doc.pages)]
    selected_ids={e['id'] for e in selected}
    decision_ids={e['id'] for e in result['sections']['decision']}
    candidates=[]
    historical_candidates=[]
    attributes=attribution_map(doc)
    marker=re.compile(r'\b(?:we\s+(?:direct|order|quash|allow|dismiss|clarify)|(?:shall|must)\s+(?:be\s+)?(?:pay|deposit|refund|release|continue|remain)|within\b.{0,45}\b(?:weeks?|months?|days?)|subject\s+to|stands?\s+(?:allowed|dismissed)|appeals?\s+succeed|larger\s+bench)\b',re.I)
    for s in doc.sentences:
        if not s['quoted'] and marker.search(s['sentence']) and s['id'] not in decision_ids:
            entry=_entry(s,'independent broad operative-language screen; attribution is heuristic')
            entry['attribution']=attributes[s['id']]
            historical_interval=elapsed_history(s['sentence'])
            if historical_interval or attributes[s['id']]['label'] in {'party_submission','reported_material','document_or_correspondence'}:
                if historical_interval:entry['screening_note']='Elapsed time in historical narration, without an obligation cue.'
                historical_candidates.append(entry)
            else:
                candidates.append(entry)
    missing=[k for k,v in result['sections'].items() if not v]
    flags=[]
    if invalid: flags.append('SOURCE_MISMATCH')
    if missing: flags.append('MISSING_SECTION')
    if candidates: flags.append('POSSIBLE_OMITTED_DIRECTIONS')
    if any(d['method']=='ocr' or not d['readable'] or d['ocr_error'] or d['replacement_characters'] for d in doc.diagnostics):
        flags.append('EXTRACTION_REVIEW')
    if result['summary_words']>result['word_limit']: flags.append('BUDGET_OVERFLOW')
    if any('No explicit issue' in w for w in result['warnings']): flags.append('ISSUE_CONTEXT_ONLY')
    if any('quotation' in w for w in result['warnings']): flags.append('QUOTE_ATTRIBUTION_REVIEW')
    if result['model'].get('mode')=='evidence_only': flags.append('MODEL_NOT_RUN')
    if result['model'].get('long_sentence_ids'): flags.append('ENCODER_TRUNCATION_REVIEW')
    return dict(status='blocked_source_mismatch' if invalid else 'review_required' if flags else 'source_checks_passed',
                flags=flags,invalid_evidence_ids=invalid,missing_sections=missing,
                exact_source_check_passed=not invalid, selected_evidence_count=len(selected),
                possible_omitted_directions=candidates, human_validation_required=True,
                reported_or_historical_direction_mentions=historical_candidates,
                attribution_method='uncalibrated text rules; not independent human attribution validation',
                legal_accuracy_measured=False,
                note='Exact source matching checks copying, not legal interpretation, attribution or completeness. Review the source and the omission candidates.')

def render_markdown(result):
    audit=result['audit']
    out=[f'# Evidence-grounded case brief — {result["source_file"]}',
         f'**Machine check:** {audit["status"]}. **Human review required.**',
         f'Words: {result["summary_words"]}; requested budget: {result["word_limit"]}.',
         'Extracts are verbatim apart from whitespace. Background can contain allegations and earlier rulings.']
    if result['caption_verbatim']:
        out+=['## Caption (verbatim; may contain multiple connected matters)',result['caption_verbatim']]
    for key,title in SECTION_TITLES.items():
        if key=='issue':
            title={'explicit_source_issue':'Issue stated in the source',
                   'source_dispute_description':'Legal dispute described in the source (no explicit question)',
                   'reported_competing_positions':'Dispute context: reported competing positions',
                   'procedural_context':'Procedural context (controlling issue not reliably identified)'}.get(result.get('issue_mode'),title)
        out.append('## '+title)
        if key=='reasoning':
            blocks={}
            for e in result['sections'][key]:blocks.setdefault(e.get('reasoning_block_id',e['id']),[]).append(e)
            for entries in blocks.values():
                pages=sorted({f['page'] for e in entries for f in e['fragments']})
                ids=', '.join(str(e['id']) for e in entries)
                out.append(f'**[pages {", ".join(map(str,pages))}; evidence {ids}]** '+' '.join(e['sentence'] for e in entries))
            if not blocks:out.append('Not reliably identified. Consult the judgment.')
            continue
        for e in result['sections'][key]:
            pages=str(e['page']) if e['page']==e['page_end'] else f'{e["page"]}–{e["page_end"]}'
            out.append(f'- **[p. {pages}; evidence {e["id"]}]** {e["sentence"]}')
        if not result['sections'][key]: out.append('Not reliably identified. Consult the judgment.')
    out+=['## Review notes']+[f'- {w}' for w in result['warnings']]
    out += ['- '+audit['note']]
    out.append('- Issue extracts may address only part of the dispute. Check that every controlling issue is covered; procedural or competing-position extracts are not court-formulated questions.')
    if audit.get('reported_or_historical_direction_mentions'):
        out.append(f'- {len(audit["reported_or_historical_direction_mentions"])} direction mentions were classified as reported material or submissions. They remain in the JSON audit for review and are not counted as current-order omissions.')
    if audit['possible_omitted_directions']:
        out+=['## Additional passages to check (may be historical; not asserted as current orders)']
        for e in audit['possible_omitted_directions']:
            out.append(f'- **[p. {e["page"]}–{e["page_end"]}; evidence {e["id"]}]** {e["sentence"]}')
    return '\n\n'.join(out)

def summarize_file(filename,budget=800,model=None,model_choice=None,ocr='auto'):
    doc=read_document(filename,ocr=ocr)
    predictions,info=({}, {'mode':'evidence_only'}) if model is None else model.score_document(doc,model_choice)
    result=build_brief(doc,budget,predictions,info)
    result['audit']=audit_brief(doc,result)
    if not result['audit']['exact_source_check_passed']:
        raise RuntimeError('Evidence verification failed. Summary withheld; inspect extraction.')
    result['markdown']=render_markdown(result)
    return result
