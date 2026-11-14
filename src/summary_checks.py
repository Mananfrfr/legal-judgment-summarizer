"""Independent rule-based screening, not an oracle for legal correctness.

No model, reference summary or outcome-selection score is consulted. Current
court attribution is still heuristic; a matched value never certifies context.
"""
import re
from decimal import Decimal, InvalidOperation
from stage44_attribution import attribution_map
from stage44_source import verify_evidence

NUMBER = r'(?:\d+(?:,\d+)*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|thirty|sixty|ninety)'
AMOUNT = re.compile(r'(?:₹|\b(?:Rs\.?|INR|rupees))\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|thousand|million)?\b', re.I)
DEADLINE = re.compile(r'\bwithin\s+(?:a\s+period\s+of\s+)?('+NUMBER+r')\s+(business\s+days?|working\s+days?|days?|weeks?|months?|years?)\b|\b(?:on\s+or\s+before|by)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})\b', re.I)
OUTCOME = re.compile(r'\b(?:appeals?|petitions?|applications?|suits?)\s+(?:(?:is|are|stands?|must\s+be)\s+)?(?:(?:hereby|accordingly|therefore)\s+)*(?P<neg>not\s+)?(?P<partial>partly\s+|partially\s+)?(?P<value>allowed|dismissed|rejected|disposed\s+of)\b|\bwe\s+(?:(?:therefore|hereby|accordingly)\s+)*(?P<direct>allow|dismiss|reject|dispose\s+of)\s+(?:the\s+|this\s+|these\s+)?(?:appeals?|petitions?|applications?|suits?)\b', re.I)
OPERATIVE = re.compile(r'\bwe\s+(?:(?:hereby|accordingly|therefore)\s+)*(?:direct|order|allow|dismiss|quash|set\s+aside|grant|reject)|\b(?:shall|must|is\s+directed\s+to|are\s+directed\s+to)\b.{0,70}\b(?:pay|deposit|refund|release|file|comply)|\b(?:subject\s+to|unless|no\s+order\s+as\s+to\s+costs)\b', re.I)
EXCLUDED = {'quoted_material','party_submission','reported_material','document_or_correspondence'}
HISTORY = re.compile(r'\b(?:had|was|were|previously|earlier)\b.{0,50}\b(?:allowed|dismissed|directed|ordered)|\b(?:High|Trial|District|lower)\s+Court\b.{0,60}\b(?:allowed|dismissed|directed|ordered)|\b(?:seeks?|prays?|requests?|submitted|argued|contended)\b', re.I)
HYPOTHETICAL = re.compile(r'\b(?:if|whether|suppose|assuming|would|could|ought|should)\b', re.I)
WORDS = dict(zip('one two three four five six seven eight nine ten eleven twelve fifteen thirty sixty ninety'.split(), [1,2,3,4,5,6,7,8,9,10,11,12,15,30,60,90]))

def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()

def extract_values(text):
    """Return explicit supported syntax only; unrecognized wording stays unknown."""
    values = []
    for match in OUTCOME.finditer(text.replace(',', ' ')):
        value = match.group('value') or {'allow':'allowed','dismiss':'dismissed','reject':'rejected','dispose of':'disposed of'}[normalize(match.group('direct'))]
        prefix = ('not ' if match.group('neg') else '') + ('partly ' if match.group('partial') else '')
        values.append(dict(kind='outcome',value=prefix+normalize(value),surface=match.group()))
    for match in AMOUNT.finditer(text):
        try:
            scale = (match.group(2) or '').lower()
            factor = 100000 if scale in {'lakh','lakhs','lac','lacs'} else 10000000 if scale in {'crore','crores'} else 1000 if scale=='thousand' else 1000000 if scale=='million' else 1
            value = format(Decimal(match.group(1).replace(',',''))*factor, 'f')
            if '.' in value:value=value.rstrip('0').rstrip('.')
        except InvalidOperation: continue
        values.append(dict(kind='amount',value='INR '+value,surface=match.group()))
    for match in DEADLINE.finditer(text):
        if match.group(3):value=normalize(match.group(3))
        else:
            n=normalize(match.group(1));n=str(WORDS.get(n,n))
            unit=normalize(match.group(2)).rstrip('s')
            value=n+' '+unit
        values.append(dict(kind='deadline',value=value,surface=match.group()))
    return values

def check_summary(doc, result):
    """Screen machine decision extracts; return evidence and explicit abstentions.

    For each selected extract use its linked source sentence, not another
    sentence containing the same value. Omission candidates are independently
    collected from unquoted operative language and remain unverified.
    """
    attributes=attribution_map(doc)
    by_id={s['id']:s for s in doc.sentences}
    source=[]
    for s in doc.sentences:
        text=s['sentence'];label=attributes[s['id']]['label']
        if label in EXCLUDED or s['quoted'] or HISTORY.search(text) or HYPOTHETICAL.search(text) or text.rstrip().endswith('?'):continue
        if OUTCOME.search(text.replace(',',' ')) or OPERATIVE.search(text):source.append(s)
    source_ids={s['id'] for s in source}
    source_values=[dict(v,source_id=s['id'],page=s['page'],source_text=s['sentence']) for s in source for v in extract_values(s['sentence'])]
    entries=result.get('sections',{}).get('decision',[])
    flags=[];claims=[];covered=set()
    same_document=result.get('source_sha256')==doc.sha256
    if not same_document:flags.append('DOCUMENT_IDENTITY_MISMATCH')
    for entry in entries:
        original=by_id.get(entry.get('id'))
        exact=bool(same_document and original and entry.get('sentence')==original['sentence'] and entry.get('fragments')==original['fragments'] and verify_evidence(entry,doc.pages))
        eligible=bool(original and original['id'] in source_ids)
        if not exact:flags.append('DECISION_SOURCE_MISMATCH')
        if not eligible:flags.append('DECISION_ATTRIBUTION_REVIEW')
        if exact and eligible:covered.add(original['id'])
        expected=extract_values(original['sentence']) if original else []
        for value in extract_values(entry.get('sentence','')):
            same=any(v['kind']==value['kind'] and v['value']==value['value'] for v in expected)
            status=('matched_linked_source' if exact and eligible and same else
                    'linked_value_absent' if original and not same else 'context_requires_review')
            claims.append(dict(value,status=status,evidence_id=entry.get('id'),
                               source_page=original['page'] if original else None,
                               source_text=original['sentence'] if original else None))
    omitted=[dict(v,status='possible_omission') for v in source_values if v['source_id'] not in covered]
    if omitted:flags.append('POSSIBLE_DECISION_DETAIL_OMISSION')
    outcomes={v['value'] for v in source_values if v['kind']=='outcome'}
    if len(outcomes)>1:flags.append('MULTIPLE_SOURCE_OUTCOMES_REVIEW')
    if not entries:flags.append('DECISION_MISSING')
    if not source:flags.append('SOURCE_OPERATIVE_PASSAGE_NOT_IDENTIFIED')
    kinds={v['kind'] for v in claims}
    if 'outcome' not in kinds:flags.append('SUMMARY_OUTCOME_NOT_RECOGNIZED')
    coverage={kind:dict(source_candidate_count=sum(v['kind']==kind for v in source_values),
                         summary_claim_count=sum(v['kind']==kind for v in claims),
                         matched_linked_source_count=sum(v['kind']==kind and v['status']=='matched_linked_source' for v in claims))
              for kind in ['outcome','amount','deadline']}
    return dict(version='1.0',status='review_required' if flags else 'no_rule_issue_detected',
                flags=list(dict.fromkeys(flags)),claims=claims,possible_omissions=omitted,counts=coverage,
                legal_correctness_verified=False,human_review_required=True,
                scope='Machine decision section only; reviewer drafts are not checked.',
                limitations=['Rule coverage is incomplete; zero extracted values is not a pass.',
                             'Current-court attribution, separate opinions and connected appeals require review.',
                             'Matching an amount or duration does not independently verify recipient, obligation, deadline start or exceptions.',
                             'Amounts recognize explicit INR/Rs/rupees/₹ numeric forms; deadlines recognize within-duration or selected numeric-date forms.'],
                note='Diagnostic source-linked checks, not a factual-accuracy score. Possible omissions may be historical or belong to other matters.')

def checks_markdown(checks):
    lines=['## Outcome, amount and deadline checks','',checks['note'],'',
           '**Status:** '+checks['status']+'. Human review required.','',
           '| Detail | Source candidates | Summary claims | Matched linked source |',
           '|---|---:|---:|---:|']
    for kind,c in checks['counts'].items():
        lines.append(f"| {kind} | {c['source_candidate_count']} | {c['summary_claim_count']} | {c['matched_linked_source_count']} |")
    lines+=['', 'Flags: '+(', '.join(checks['flags']) or 'No rule issue detected; correctness is not certified.'),'']
    for c in checks['claims']:
        lines.append(f"- {c['kind']}: {c['value']} — {c['status']}; evidence {c['evidence_id']}, source page {c['source_page']}.")
    for c in checks['possible_omissions']:
        lines.append(f"- Possible omitted {c['kind']}: {c['value']}; source evidence {c['source_id']}, page {c['page']}: {c['source_text']}")
    return '\n'.join(lines)
