"""Conservative attribution cues, separate from learned role/importance scores.

Labels describe textual evidence, not calibrated probabilities or verified legal
roles. Paragraph context prevents a later sentence in a submission/letter from
being silently promoted into the current court's reasoning.
"""
import re
from stage44_source import quote_mask

PARTY = re.compile(r'\b(?:(?:(?:learned\s+)?(?:senior\s+)?(?:counsel|ASG)|(?:Mr\.|Ms\.|Shri)\s+[\w. ]{1,45}|(?:the\s+)?(?:appellant|respondent|petitioner|claimant)(?:[\s.-]+(?:no\.?\s*)?\d+)?)\s+(?:(?:also|further|strongly|on\s+the\s+other\s+hand)\s+|,\s*)*(?:submitted?|submits|contends?|contended|argued|urged|prayed|pointed\s+out|invited\s+our\s+attention|referred|refuted)|according\s+to\s+(?:learned|Mr\.|Ms\.|Shri|the\s+(?:appellant|respondent|petitioner))|it\s+was\s+(?:(?:also|further)\s+)?(?:submitted|argued|contended|canvassed|pointed\s+out))\b', re.I)
REPORTED = re.compile(r'\b(?:(?:High|Trial|Family|District|appellate|Executing)\s+Court\b.{0,65}\b(?:held|found|observed|directed|dismissed|allowed|opined|ordered)|(?:Majority\s+Award|arbitral\s+tribunal|earlier\s+(?:court|judgment))\b.{0,60}\b(?:held|found|observed|directed)|(?:complaint|FIR|affidavit)\s+(?:alleges|states|refers)|(?:petitioner|appellant|respondent|complainant)\s+alleges)\b', re.I)
MATERIAL = re.compile(r'\b(?:(?:following|said)\s+(?:letter|correspondence)|letter\s+(?:was\s+)?written|(?:letter|agreement|affidavit|government\s+order)\b.{0,65}\b(?:reads|states|extracted|reproduced)|Dear\s+Sirs|Yours\s+faithfully)\b', re.I)
COURT = re.compile(r'\b(?:we\s+(?:(?:are|have|do|would|cannot|can\s+not|must|find|hold|agree|disagree|consider|notice|observe|note|think|conclude|direct|allow|dismiss|quash|set\s+aside|clarify|grant|restore|reject|invoke|make\s+it\s+clear))|in\s+our\s+(?:view|opinion)|our\s+(?:prima\s+facie\s+)?(?:conclusion|view)|this\s+Court\s+finds|let\s+us\s+(?:first\s+)?examine)\b', re.I)
ASSESSMENT = re.compile(r'\b(?:from\s+a\s+prima\s+facie\s+examination|going\s+by\s+the\s+binding\s+precedents|it\s+(?:is|would\s+be)\s+(?:clear|evident|settled|trite)|it\s+goes\s+without\s+saying|(?:this|such|the\s+above)\s+(?:reasoning|observation|exercise)\b.{0,110}\b(?:conjecture|unsustainable|not\s+been\s+carried|unknown\s+to)|(?:a\s+bare\s+perusal|a\s+reading\s+of)\b.{0,100}\b(?:indicate|apparent)|claim\b.{0,40}\bbarred\s+by\s+limitation|no\s+enforceable\s+legal\s+right|prima\s+facie\s+not\s+made\s+out|such\s+a\s+claim\s+discloses|the\s+aforesaid\s+instructions\s+make\s+it\s+clear)\b', re.I)
LEGAL_DISPUTE = re.compile(r'\b(?:on\s+the\s+ground\b.{0,200}\b(?:lis|dispute|challenge)|(?:lis|dispute|controversy)\b.{0,40}\b(?:arisen|concerns|relates)|(?:question|issue)\b.{0,65}\b(?:before\s+us|consider|examine|determine))\b', re.I)
HEARING = re.compile(r'^(?:We\s+have\s+(?:also\s+|carefully\s+)?(?:heard|considered\s+the\s+submissions)|(?:Elaborate\s+)?arguments\s+were\s+heard|As\s+such,\s+we\s+have\s+disposed)', re.I)
ASSESSMENT_EXTRA = re.compile(r'\b(?:claim\s+stands?\s+barred|(?:reasoning|finding|direction|judgment|order)\b.{0,100}\b(?:unsustainable|conjecture|oblivious|belied)|(?:mere|merely)\s+(?:conjecture|presumption)|no\s+meaningful\s+discussion\s+on\s+merits)\b', re.I)
CITATION_INTRO = re.compile(r'\b(?:we\s+(?:would\s+like\s+to\s+)?(?:reproduce|quote|extract)|it\s+would\s+be\s+beneficial\s+to\s+reproduce)\b',re.I)
CURRENT_THIRD_PERSON=re.compile(r'\bthis\s+Court\s+(?:is\s+of\s+the\s+(?:considered\s+)?view|holds|reiterates|concludes|finds)\b',re.I)

def prior_authority(text):
    return bool(re.match(r'^In\s+(?!our\b|the\s+present\b|this\b|view\b|light\b|these\b|such\b)',text,re.I)
                and re.search(r'\b(?:v\.|vs\.|supra|SCC)',text[:180],re.I)
                and re.search(r'\b(?:held|was|explained|laid\s+down|considered|decided)\b',text,re.I))

def voice_text(text):
    mask=quote_mask(text)
    return ''.join(' ' if quoted else char for char,quoted in zip(text,mask))

def assessment(text):
    unquoted=voice_text(text)
    return not HEARING.search(unquoted) and not prior_authority(unquoted) and bool(COURT.search(unquoted) or CURRENT_THIRD_PERSON.search(unquoted) or ASSESSMENT.search(unquoted) or ASSESSMENT_EXTRA.search(unquoted))

def attribution_map(doc):
    result = {}
    for paragraph in doc.paragraphs:
        members = [s for s in doc.sentences if s['paragraph_id'] == paragraph['id']]
        text = paragraph['text']
        first = members[0]['sentence'] if members else ''
        party_context = bool(PARTY.search(first))
        material_context = bool(MATERIAL.search(text[:500]))
        precedent_context=prior_authority(first)
        submission_section=bool(re.search(r'\b(?:ARGUMENTS|SUBMISSIONS|CONTENTIONS)\b',paragraph.get('source_section',''),re.I))
        if submission_section and re.match(r'^(?:He|She|They|According\s+to\s+(?:him|her|them))\b',first,re.I):party_context=True
        # A court introduction establishes context, but not a label for every
        # allegation or authority subsequently reported in that paragraph.
        court_context = assessment(first) and not party_context and not material_context
        for s in members:
            t = s['sentence']
            if s['quoted']:
                label, cue = 'quoted_material', 'quotation span; speaker is not assumed to be the present court'
            elif precedent_context:
                label, cue = 'reported_material', 'paragraph reports a named prior authority; not the present controlling issue'
            elif t.isupper() and len(re.sub(r'[^A-Z]','',t))>8 and not t.rstrip().endswith(('.', '?', '!')):
                label, cue = 'heading_fragment', 'uppercase heading fragment, not a reasoning sentence'
            elif CITATION_INTRO.search(t) and t.rstrip().endswith((':',':-')):
                label, cue = 'citation_introduction', 'introduces reproduced material rather than giving a reason'
            elif material_context:
                label, cue = 'document_or_correspondence', 'paragraph introduces a letter or reproduced document'
            elif PARTY.search(t) or party_context:
                # An explicit new "we" sentence can return to court voice, but
                # not when embedded inside the same party submission sentence.
                if not PARTY.search(t) and re.match(r'^(?:However,\s*)?We\s+', t, re.I):
                    party_context = False
                    court_context = True
                    label, cue = 'current_court', 'explicit return to court voice'
                else:
                    label, cue = 'party_submission', 'reported submission or its paragraph context'
            elif HEARING.search(t):
                label, cue = 'court_procedure', 'hearing or filing history; not substantive reasoning'
            elif assessment(t):
                label, cue = 'current_court', 'present court assessment cue'
                court_context = True
            elif REPORTED.search(t):
                label, cue = 'reported_material', 'earlier ruling, allegation or external finding'
            elif court_context:
                label, cue = 'court_context', 'consecutive context within a court-assessment paragraph'
            else:
                label, cue = 'unresolved', 'no sufficiently specific attribution cue'
            result[s['id']] = dict(label=label, cue=cue, method='textual_rules', verified=False)
        # Preserve preceding context within the same paragraph. Never promote
        # quoted/attributed submissions or earlier rulings by this backfill.
        if any(result[s['id']]['label']=='current_court' for s in members) and not party_context and not material_context:
            for s in members:
                if result[s['id']]['label']=='unresolved':
                    result[s['id']]=dict(label='court_context', cue='context in a paragraph containing court assessment; not a separate factual finding',method='textual_rules',verified=False)
    return result

def current_analysis(s, attributes):
    return attributes[s['id']]['label'] in {'current_court', 'court_context'}
