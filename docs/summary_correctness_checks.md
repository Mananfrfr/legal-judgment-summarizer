# Outcome, amount and deadline screening

The app now adds an independent rule-based screen to every machine brief. It runs locally, needs no API key, changes no model weights and does not select or rewrite summary sentences.

## What it does

1. Read source sentences and their page/span references.
2. Independently identify candidate operative language. Exclude recognized quotations, party submissions, reports of earlier rulings, hypothetical wording and questions. This attribution is heuristic, not a gold label.
3. Extract explicit outcomes, INR amounts and deadlines from those passages and from the machine decision section.
4. For each summary claim, compare it with **its linked source sentence**. A matching number elsewhere in the document cannot validate a changed claim.
5. Show matched values, unsupported linked values, context warnings and possible omitted source details, with page references and source text.

The app displays the checks in a dedicated card. The same report appears in `audit.decision_detail_checks` in JSON and in the Markdown/ZIP exports. Editing a reviewer draft does not update or inherit these machine checks; the UI explicitly labels that boundary.

## How to read the results

| Status | Meaning |
|---|---|
| `matched_linked_source` | The value occurs in the linked source sentence, the full extract/spans match, and the source sentence passed the operative-language screen. Attribution still requires review. |
| `linked_value_absent` | The recognized value is absent from the linked source sentence. Review the claim and its reference. |
| `context_requires_review` | A value match is insufficient: text/spans changed or the operative attribution was not established. |
| `possible_omission` | A candidate source detail was not retained as a complete linked decision extract. It may belong to history or another matter. |
| `no_rule_issue_detected` | The implemented rules found no issue. This is **not** a finding that the entire summary is correct. |

Example: source says “We direct the insurer to pay Rs. 2,00,000 within six weeks from today, subject to adjustment of earlier payments.” A changed amount or duration is flagged against that sentence. Removing the adjustment condition produces a source-context mismatch even if the amount and duration still match. Omitting the entire direction produces possible amount/deadline omissions.

## Recognized syntax and limitations

- Outcomes: explicit allowed/dismissed/rejected/disposed-of forms and selected “we allow/dismiss…” forms. Partial outcomes and negation remain distinct. Multiple detected outcomes trigger review; they may legitimately concern separate appeals.
- Amounts: numeric `Rs`, `INR`, `rupees` or `₹` forms, including Indian commas, decimals, lakh/lac, crore, thousand and million. `Rs. 2,00,000` and `INR 2 lakh` normalize to the same value. Other currencies, spelled-out monetary values, interest rates and all financial conditions are not comprehensively parsed.
- Deadlines: `within` a numeric or supported small word-number duration; business days remain distinct from ordinary days. Selected `by` / `on or before` date forms are recognized. Months are not converted to fixed days. Alternative date formats are not semantically reconciled.
- The full source sentence is shown because the recipient, obligation, deadline starting event, exceptions and qualifications matter. Matching a number alone does not validate these relationships.
- Unrecognized outcomes, missing decisions and absent operative candidates produce review flags. No extracted claims is never treated as evidence of correctness.
- Rules inspect the **machine decision section**, not all facts and reasoning or free-form reviewer drafts. Source identity and linked evidence are checked. A false attribution by the source parser can still lead to a misleading match.
- Candidate omissions are not reference annotations. Do not calculate factual precision/recall by treating these candidates as ground truth.

## Run against a saved machine summary

```bash
python scripts/check_summary_details.py --source judgment.pdf --summary summary.json --output outputs/detail_check --ocr off
```

Use the original source and JSON export with evidence IDs and spans. The command writes `detail_checks.json` and `detail_checks.md`. It does not require trained weights. The regular app/CLI runs these checks automatically as part of the audit.

## Faculty demonstration

```bash
python scripts/demo_summary_checks.py
```

This uses an invented judgment, creates a matching summary, deliberately changes outcome/amount/deadline fields and removes a direction, and prints the resulting flags. The demonstration is a controlled test, not a measured accuracy on real judgments.

For a factual-correctness F1 score, first obtain independently reviewed labels for decision facts and errors. Evaluate these rules against those labels and report false positives, false negatives and abstentions. An LLM judge is not included in this change.
