"""Audited annotation policy from the original Stage 2.2 experiment."""
import hashlib
import pandas as pd
LABELS = ['ANALYSIS','ARG_PETITIONER','ARG_RESPONDENT','FAC','ISSUE','NONE','PREAMBLE','PRE_NOT_RELIED','PRE_RELIED','RATIO','RLC','RPC','STA']
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))


def json_to_df(documents, split):
        rows, skipped, multiple_sets = [], [], []
        for doc_index, doc in enumerate(documents):
            annotations = doc.get('annotations') or []
            # A document may contain labels from several annotators. Never merge their
            # results: this would duplicate offsets and may invent contradictory labels.
            candidates = []
            for annotation_index, annotation in enumerate(annotations):
                if not isinstance(annotation, dict):
                    continue
                results = annotation.get('result') or []
                spans_here = [r['value'] for r in results if isinstance(r, dict)
                              and isinstance(r.get('value'), dict)
                              and 'text' in r['value']]
                if spans_here:
                    candidates.append((annotation_index, spans_here, annotation))

            if not candidates:
                skipped.append(doc_index)
                continue

            # Prefer an explicitly adjudicated/ground-truth annotation, if supplied.
            # The Hugging Face Parquet schema often contains only 'result' and lacks
            # adjudication metadata; in that case, use a documented deterministic policy.
            adjudicated = [item for item in candidates if
                           item[2].get('ground_truth') is True or
                           item[2].get('is_adjudicated') is True]
            if len(adjudicated) > 1:
                raise ValueError(f'{split} document {doc_index}: more than one annotation '
                                 'is marked adjudicated; review the source data.')
            selected_idx, spans, _ = (adjudicated[0] if adjudicated else candidates[0])
            selection_reason = 'adjudicated' if adjudicated else 'first_nonempty'

            if len(annotations) > 1:
                # Compare the entire labeled sentence sequence across nonempty sets.
                # Keep only counts/indices in audit, never duplicate the document text.
                def signature(annotator_spans):
                    records = [(v.get('start'), v.get('end'), v.get('text'),
                                tuple(v.get('labels') or ())) for v in annotator_spans]
                    return tuple(sorted(records, key=lambda x: (str(x[0]), str(x[1]), str(x[2]))))
                disagree = len({signature(item[1]) for item in candidates}) > 1
                multiple_sets.append(dict(doc_index=doc_index,
                                          total_annotation_sets=len(annotations),
                                          nonempty_annotation_sets=len(candidates),
                                          selected_annotation_index=selected_idx,
                                          selection_reason=selection_reason,
                                          annotation_sets_disagree=disagree))
            for value in spans:
                if not isinstance(value.get('start'), int) or not isinstance(value.get('end'), int):
                    raise ValueError(f'{split} document {doc_index}: missing sentence offsets')
                if value['end'] <= value['start'] or not value['text'].strip():
                    raise ValueError(f'{split} document {doc_index}: invalid or empty span')
                labels = value.get('labels') or []
                if len(labels) > 1:
                    raise ValueError('Expected one rhetorical role per sentence')
                label = labels[0] if labels else None
                if label is not None and label not in LABEL2ID:
                    raise ValueError(f'Unknown label: {label}')
                rows.append(dict(split=split, doc_index=doc_index, annotation_index=selected_idx, meta=doc.get('meta'),
                                 start=value['start'], end=value['end'], text=value['text'], label=label))
        df = pd.DataFrame(rows, columns=['split','doc_index','annotation_index','meta','start','end','text','label'])
        df = df.sort_values(['doc_index', 'start', 'end']).reset_index(drop=True)
        if df.duplicated(['doc_index', 'start', 'end']).any():
            raise ValueError(f'{split}: duplicate sentence spans')
        df['sentence_index'] = df.groupby('doc_index').cumcount()
        report = dict(raw_documents=len(documents), represented_documents=int(df.doc_index.nunique()),
                      sentences=len(df), skipped_document_indices=skipped,
                      unlabeled=int(df.label.isna().sum()),
                      multi_annotation_documents=len(multiple_sets),
                      multi_annotation_disagreements=sum(x['annotation_sets_disagree'] for x in multiple_sets),
                      annotation_selection_policy='adjudicated_if_marked_else_first_nonempty',
                      multi_annotation_details=multiple_sets)
        return df, report

def document_hashes(df):
        return {hashlib.sha256(' '.join(' '.join(g.text).split()).encode()).hexdigest()
                for _, g in df.groupby('doc_index')}
