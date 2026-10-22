"""Page-addressable, non-generative extraction. Offsets refer to exported canonical_pages.

Only whitespace is normalized. Never repair names, currencies, numbers or negation.
Evidence remains independently checkable even when the source PDF needs OCR.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import re
import fitz

def norm(text):
    return re.sub(r'\s+', ' ', text).strip()

ROMAN_SECTIONS={value:i for i,value in enumerate(('I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX'),1)}

@dataclass
class Document:
    source_file: str
    sha256: str
    pages: list
    paragraphs: list
    sentences: list
    diagnostics: list
    warnings: list
    caption: str

    def to_dict(self):
        return asdict(self)

HEADING = re.compile(r'^(?:J\s*U\s*D\s*G\s*(?:E\s*)?M\s*E\s*N\s*T|O\s*R\s*D\s*E\s*R)\s*$', re.I)
NUMBER = re.compile(r'^[‘“"\s]*(\d{1,3}(?:\.\d{1,2}){0,2})(?:[.)])(?:\s+|$)')
FURNITURE = re.compile(r'^(?:Page\s+\d+\s+of\s+\d+|\d{1,3}|Signature Not Verified|Digitally signed by.*|Date:\s*\d.*|Reason:|\d{2}:\d{2}:\d{2}.*)$', re.I)
CASE_FOOTER = re.compile(r'^(?:Civil|Criminal|Arbitration|Writ|Review|Contempt|Special Leave|SLP|S\.?L\.?P\.?)\s*.*(?:No\.|Nos\.|of\s+20\d\d)', re.I)
SIGNED = re.compile(r'^(?:[.\s…\uFFFD]{5,}.*\b(?:J|CJI)\.|NEW DELHI\s*[;,]?$)', re.I)
ABBREVIATION = re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Smt|Shri|Rs|No|Nos|Co|Ltd|Inc|vs|v|Hon|Art|Sec|para|pp|ibid|Ors|Anr|Jr|Sr|Crl|Civ|Govt|Dept)\.$', re.I)

def sentence_ranges(text):
    """Boundaries keep decimal amounts, legal abbreviations and initials intact."""
    start = 0
    for m in re.finditer(r'[.!?][”"\u2019\)]*\s+', text):
        p = m.start()
        prefix = text[max(0, p-25):p+1]
        if text[p]=='.' and re.fullmatch(r'(?:ii|iii|iv|vi|vii|viii|ix|xi|xii)\.',text[start:p+1].strip(),re.I):
            continue # A roman list marker starts an item, never a sentence alone.
        # Protect the abbreviation only in its legal-code context. A global
        # "Cr." exception would also join unrelated sentences ending in crore.
        if text[p] == '.' and re.search(r'\bCr\.$', prefix, re.I) and re.match(r'\s*P\.?\s*C\b', text[p+1:], re.I):
            continue
        if text[p] == '.' and (ABBREVIATION.search(prefix) or re.search(r'\b(?:[A-Za-z]\.)+$', prefix)):
            continue
        end = m.end()
        yield start, m.start()+len(m.group().rstrip())
        start = end
    if start < len(text):
        yield start, len(text)

def cut_fragments(fragments, start, end):
    out, offset = [], 0
    for f in fragments:
        a, b = max(start, offset), min(end, offset+len(f['text']))
        if a < b:
            x, y = f['start']+a-offset, f['start']+b-offset
            out.append(dict(page=f['page'], start=x, end=y, text=f['text'][a-offset:b-offset]))
        offset += len(f['text'])+1
    return out

def advance_quotes(text,stack):
    """Track quote families consistently; apostrophes within words are not quotes."""
    for i,c in enumerate(text):
        if c in '‘“':stack.append('single' if c=='‘' else 'double')
        elif c in '’”':
            if c=='’' and i and i+1<len(text) and text[i-1].isalpha() and text[i+1].isalpha():continue
            family='single' if c=='’' else 'double'
            if family in stack:del stack[len(stack)-1-stack[::-1].index(family):]
        elif c=='"':
            if stack and stack[-1]=='double':stack.pop()
            else:stack.append('double')

def quote_mask(text):
    stack,mask=[],[]
    for i,c in enumerate(text):
        mask.append(bool(stack))
        if c=='’' and i and i+1<len(text) and text[i-1].isalpha() and text[i+1].isalpha():continue
        advance_quotes(c,stack)
    return mask

def _native_lines(page):
    lines = []
    for block in page.get_text('dict', sort=True)['blocks']:
        for line in block.get('lines', []):
            raw=''.join(s['text'] for s in line['spans'])
            mask=[]
            for span in line['spans']:
                superscript=bool(span['flags'] & 1) and span['text'].strip().isdigit()
                mask.extend([superscript]*len(span['text']))
            text, normalized_mask='',[]
            for token in re.finditer(r'\S+',raw):
                if text:
                    text+=' '; normalized_mask.append(False)
                text+=token.group(); normalized_mask.extend(mask[token.start():token.end()])
            skips=[]
            for match in re.finditer('1+', ''.join('1' if x else '0' for x in normalized_mask)):
                skips.append((match.start(),match.end()))
            if text:
                lines.append(dict(text=text, x=line['bbox'][0], y=line['bbox'][1],
                                  size=max(s['size'] for s in line['spans']),skip_ranges=skips))
    lines=sorted(lines, key=lambda x: (round(x['y'], 1), x['x']))
    joined=[]
    for line in lines:
        if joined and abs(joined[-1]['y']-line['y'])<.8 and abs(joined[-1]['size']-line['size'])<1:
            shift=len(joined[-1]['text'])+1
            joined[-1]['skip_ranges'].extend((a+shift,b+shift) for a,b in line['skip_ranges'])
            joined[-1]['text']+=' '+line['text']
        else:
            joined.append(dict(line))
    return joined

def read_document(filename, ocr='auto', max_pages=200):
    path = Path(filename)
    if ocr not in {'auto','off','force'}:
        raise ValueError('OCR mode must be auto, off or force.')
    if path.stat().st_size > 50_000_000:
        raise ValueError('Input exceeds the 50 MB limit. Split the document first.')
    diagnostics, warnings, page_lines = [], [], []
    if path.suffix.lower() == '.pdf':
        with fitz.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError('Unlock the password-protected PDF before processing.')
            if len(pdf) > max_pages:
                raise ValueError(f'Input exceeds {max_pages} pages; no pages were silently discarded.')
            for pi, page in enumerate(pdf, 1):
                lines = _native_lines(page)
                native = ' '.join(x['text'] for x in lines)
                suspect = len(re.findall(r'[A-Za-z]{2,}', native)) < 15 or native.count('\uFFFD') > 8
                method, error = 'native', None
                if ocr == 'force' or (ocr == 'auto' and suspect):
                    try:
                        import pytesseract
                        from PIL import Image
                        pix = page.get_pixmap(dpi=200)
                        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                        data = pytesseract.image_to_data(image, lang='eng', config='--psm 3',
                                                        timeout=60, output_type=pytesseract.Output.DICT)
                        groups = {}
                        for i, word in enumerate(data['text']):
                            if word.strip():
                                key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
                                groups.setdefault(key, []).append(i)
                        candidate = [dict(text=norm(' '.join(data['text'][i] for i in ids)),
                                          x=data['left'][ids[0]]*72/200, y=data['top'][ids[0]]*72/200,
                                          size=12) for ids in groups.values()]
                        if sum(len(x['text']) for x in candidate) > 60:
                            lines, method = candidate, 'ocr'
                            warnings.append(f'Page {pi}: OCR text requires visual verification.')
                        else:
                            error = 'OCR produced insufficient text'
                    except Exception as exc:
                        error = type(exc).__name__
                text = ' '.join(x['text'] for x in lines)
                readable = len(re.findall(r'[A-Za-z]{2,}', text)) >= 10
                diagnostics.append(dict(page=pi, method=method, characters=len(text), readable=readable,
                                        replacement_characters=text.count('\uFFFD'), ocr_error=error))
                if not readable or error:
                    warnings.append(f'Page {pi}: extraction needs review ({error or "sparse/unreadable text"}).')
                # Remove individual signature metadata lines only, not a body-containing block.
                page_lines.append((lines, page.rect.height))
    elif path.suffix.lower() == '.txt':
        text = path.read_text(encoding='utf-8-sig', errors='strict')
        lines = [dict(text=norm(t),x=0,y=i*16,size=12) for i,t in enumerate(text.splitlines()) if norm(t)]
        page_lines = [(lines, max(100, len(lines)*16))]
        diagnostics = [dict(page=1, method='text',characters=len(text),readable=bool(text.strip()),replacement_characters=text.count('\uFFFD'),ocr_error=None)]
    else:
        raise ValueError('Use an English judgment in PDF or UTF-8 TXT format.')
    return assemble(path.name, hashlib.sha256(path.read_bytes()).hexdigest(), page_lines, diagnostics, warnings)

def assemble(name, digest, page_lines, diagnostics, warnings):
    pages, paragraphs, caption = [], [], []
    current, signed, appendix, quote_depth = None, False, False, 0
    quote_stack = []
    quote_origin_x=None
    quote_origin_inline=False
    outline_headings={}
    recovered_heading_pending=False
    native_geometry=bool(diagnostics) and all(d.get('method')=='native' for d in diagnostics)
    recovered_boundary=False
    source_section=''
    first_lines = page_lines[0][0] if page_lines else []
    has_heading = any(HEADING.match(x['text']) for x in first_lines)
    body = not has_heading
    global_sizes=Counter()
    for lines,_ in page_lines:
        for line in lines:
            global_sizes[round(line['size'])]+=len(line['text'])
    global_body_size=global_sizes.most_common(1)[0][0] if global_sizes else 12
    body_lefts=sorted(line['x'] for lines,_ in page_lines for line in lines
                      if round(line['size'])>=global_body_size-1 and len(line['text'])>25)
    body_left=body_lefts[len(body_lefts)//10] if body_lefts else 0

    def flush():
        nonlocal current
        if current and current['fragments']:
            current['text'] = ' '.join(f['text'] for f in current['fragments'])
            current['id'] = len(paragraphs)
            current['page'] = current['fragments'][0]['page']
            current['page_end'] = current['fragments'][-1]['page']
            paragraphs.append(current)
        current = None

    for pi, (lines, height) in enumerate(page_lines, 1):
        canonical = ' '.join(x['text'] for x in lines)
        pages.append(canonical)
        offset = 0
        # Native geometry separates footnotes/signature overlays without deleting body text near margins.
        sizes = Counter()
        for line in lines:
            sizes[round(line['size'])] += len(line['text'])
        body_size = sizes.most_common(1)[0][0] if sizes else 12
        for line in lines:
            text, start = line['text'], offset
            offset += len(text)+1
            if re.search(r'R\s*E\s*C\s*O\s*R\s*D\s+O\s*F\s+P\s*R\s*O\s*C\s*E\s*E\s*D\s*I\s*N\s*G\s*S', text, re.I):
                flush(); appendix = True
            if appendix or signed:
                continue
            if pi == 1 and not body:
                if HEADING.match(text):
                    body = True
                else:
                    caption.append(text)
                continue
            if HEADING.match(text):
                flush(); continue
            if SIGNED.match(text):
                # A quoted earlier order can have its own signature half-way through a judgment.
                if pi >= len(page_lines)-1 and not quote_stack:
                    flush(); signed = True
                continue
            if FURNITURE.match(text):
                continue
            if round(line['size']) <= global_body_size-3:
                continue
            if line['y'] > height*.88 and CASE_FOOTER.match(text):
                continue
            if re.fullmatch(r'[A-Z .\-]+,?\s+J\.?', text) and not paragraphs and current is None:
                continue
            # An unclosed inline quotation must not swallow a structurally
            # established final section. Require an existing Roman outline,
            # matching font/margin and forward numbering; block quotes stay quoted.
            outline=re.match(r'^([IVX]+)\.\s+([A-Z].*)$',text) if text.isupper() else None
            if outline and outline.group(1) in ROMAN_SECTIONS:
                number=ROMAN_SECTIONS[outline.group(1)]
                layout=(round(line['x']/3),round(line['size']))
                known=outline_headings.get(layout,set())
                final_heading=bool(re.fullmatch(r'(?:CONCLUSION|CONCLUSIONS|FINAL ORDER|DISPOSITION)\s*:?',outline.group(2)))
                if (native_geometry and quote_stack and quote_origin_inline and final_heading and {1,2}.issubset(known)
                        and len(known)>=3 and number>max(known)):
                    flush();quote_stack.clear();quote_origin_x=None;quote_origin_inline=False
                    recovered_heading_pending=True
                    warnings.append(f'Page {pi}: unclosed inline quotation recovered at a matching numbered final-section heading; verify attribution.')
                if not quote_stack:
                    outline_headings.setdefault(layout,set()).add(number)
            if re.match(r'^(?:FACTS|ANALYSIS|REASONS|REASONING|CONCLUSION|POST-SCRIPT|ISSUES|SUBMISSIONS)\s*:?$', text, re.I):
                flush();source_section=text; continue
            if text.isupper() and 2<=len(text.split())<=14 and re.search(r'[A-Z]{3}',text) and not re.search(r'[.!?;]$',text) and not NUMBER.match(text):
                flush();source_section=text;continue
            if re.fullmatch(r'[A-Z][A-Z /,()0-9-]{4,}\s*:\s*-?',text):
                flush(); continue
            closing = {'‘':'’','“':'”','"':'"'}.get(text[:1])
            num = NUMBER.match(text)
            # Recover only at a numbered, unquoted paragraph returning to the
            # body margin from an indented quotation. Never reset by length or
            # merely because a passage sounds like an order.
            recovered_boundary=False
            if quote_stack and num and not closing and quote_origin_x is not None and line['x']+12<quote_origin_x and line['x']<=body_left+8:
                flush();quote_stack.clear();quote_origin_x=None;quote_origin_inline=False
                recovered_boundary=True
                warnings.append(f'Page {pi}, paragraph {num.group(1)}: quotation boundary recovered from PDF indentation; verify attribution.')
            previous_text = current['fragments'][-1]['text'] if current and current['fragments'] else ''
            opens_block = bool(closing) and not quote_stack and (bool(NUMBER.match(text)) or previous_text.endswith(':'))
            if opens_block:
                flush()
            num = NUMBER.match(text)
            # Numbered paragraphs must not be merged into giant page-long blocks.
            if num:
                flush()
            elif current and current['number'] is None and current['fragments'][-1]['text'].endswith(('.', '?', '!')):
                flush()
            if current is None:
                current = dict(fragments=[], number=num.group(1) if num else None,
                               quoted=bool(quote_stack) or opens_block, quote_uncertain=recovered_boundary or recovered_heading_pending,
                               source_section=source_section)
                recovered_heading_pending=False
            if num:
                start += num.end()
                text = text[num.end():]
            line_offset=offset-len(line['text'])-1
            spans=[(start-line_offset,len(line['text']))]
            for skip_a,skip_b in line.get('skip_ranges',[]):
                parts=[]
                for a,b in spans:
                    if skip_b<=a or skip_a>=b: parts.append((a,b))
                    else:
                        if a<skip_a: parts.append((a,skip_a))
                        if skip_b<b: parts.append((skip_b,b))
                spans=parts
            for a,b in spans:
                piece=line['text'][a:b]
                if not piece.strip(): continue
                left=len(piece)-len(piece.lstrip())
                right=len(piece.rstrip())
                piece_start=line_offset+a+left
                fragment = dict(page=pi, start=piece_start, end=line_offset+a+right, text=piece.strip())
                # Adjacent lines form one verifiable page span.
                fs = current['fragments']
                if fs and fs[-1]['page']==pi and canonical[fs[-1]['end']:piece_start]==' ':
                    fs[-1]['end']=fragment['end']; fs[-1]['text']+=' '+fragment['text']
                else:
                    fs.append(fragment)
            raw = line['text']
            was_quoted=bool(quote_stack)
            advance_quotes(raw,quote_stack)
            if not was_quoted and quote_stack:
                quote_origin_x=line['x'];quote_origin_inline=not opens_block
            elif not quote_stack:
                quote_origin_x=None;quote_origin_inline=False
            if current and current['quoted'] and not quote_stack:
                flush()
    flush()
    if sum(len(p) for p in pages) > 2_000_000:
        raise ValueError('Document exceeds 2 million characters; split it before processing.')
    if not paragraphs:
        raise ValueError('No readable judgment body found. Check the PDF or use OCR.')
    if quote_stack:
        warnings.append('Unbalanced quotation marks: verify quoted-precedent attribution.')
    sentences = []
    for p in paragraphs:
        mask=quote_mask(p['text'])
        attributed=bool(re.search(r'\b(?:held|observed|quoted|reproduce|as follows|said|letter|correspondence|findings|submission)\b.{0,160}(?::|[“\"])',p['text'],re.I))
        for a,b in sentence_ranges(p['text']):
            fs = cut_fragments(p['fragments'],a,b)
            text = ' '.join(f['text'] for f in fs)
            if not text.strip():
                continue
            quoted_chars=sum(mask[a:b])
            substantial_quote=quoted_chars >= 40 and quoted_chars/max(1,b-a) >= .45
            sentences.append(dict(id=len(sentences), paragraph_id=p['id'], sentence=text, fragments=fs,
                                  page=fs[0]['page'],page_end=fs[-1]['page'],
                                  quote_boundary_recovered=p['quote_uncertain'],
                                  source_section=p['source_section'],
                                  quoted=p['quoted'] or substantial_quote or (attributed and quoted_chars > 0)))
    return Document(name,digest,pages,paragraphs,sentences,diagnostics,list(dict.fromkeys(warnings)), ' '.join(caption))

def verify_evidence(entry, pages):
    fs = entry.get('fragments', [])
    if not fs or entry.get('sentence') != ' '.join(f.get('text','') for f in fs):
        return False
    if entry.get('page') != fs[0]['page'] or entry.get('page_end') != fs[-1]['page']:
        return False
    for f in fs:
        if not (1 <= f['page'] <= len(pages) and 0 <= f['start'] < f['end'] <= len(pages[f['page']-1])):
            return False
        if pages[f['page']-1][f['start']:f['end']] != f['text']:
            return False
    return True
