"""Generate four concise presentation notebooks from the project workflow."""
from pathlib import Path
import json,textwrap
ROOT=Path(__file__).resolve().parents[1]

def md(text):return dict(cell_type='markdown',metadata={},source=textwrap.dedent(text).strip().splitlines(True))
def code(text,project=True):
    text=textwrap.dedent(text).strip()+'\n'
    return dict(cell_type='code',metadata={},source=(('%%project\n' if project else '')+text).splitlines(True),execution_count=None,outputs=[])

setup=[md('''## Setup — run once
The project uses an isolated Python environment so its Hugging Face packages do not conflict with Colab's installed Gradio/Diffusers.
`%%project` means “run this ordinary Python cell in that environment.” Variables persist between these cells.
The first setup downloads dependencies and can take several minutes. No token is needed for the demo or downloaded trained model.
For the gated role dataset, accept its access conditions and add `HF_TOKEN` in Colab Secrets; never paste it into a cell.
'''),code('''from pathlib import Path
import os, subprocess, sys
ROOT = Path(os.environ.get('PROJECT_ROOT', '/content/legal-judgment-summarizer' if Path('/content').exists() else Path.cwd()))
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists():
    ROOT = ROOT.parent
if not (ROOT / 'src').exists():
    subprocess.check_call(['git', 'clone', 'https://github.com/Mananfrfr/legal-judgment-summarizer.git', str(ROOT)])
sys.path.insert(0, str(ROOT / 'src'))
from notebook_runtime import setup
setup(ROOT)
''',False),code('''from pathlib import Path
import os, json
import pandas as pd
from IPython.display import display, Markdown
ROOT = Path(os.environ['PROJECT_ROOT'])
''')]

one=[md('''# 01 · Data and preprocessing
**Question:** How do we turn a judgment into reliable training examples?

**Input:** OpenNyAI judgments with sentence spans and 13 rhetorical roles. **Output:** one ordered row per sentence, official splits, and an audit report.
This notebook has a small invented demo for tomorrow's presentation. Demo scores are not project results. Set `DEMO=False` to prepare the real data.
'''),*setup,
md('''## 1. Load the data
OpenNyAI teaches sentence roles (facts, arguments, reasoning, rulings). IN-Abs is a separate dataset of judgments paired with summaries, used later for sentence selection. They solve different tasks.
The role dataset is pinned to the revision used by the recorded experiments. We keep the official document splits; random sentence splitting could leak the same case into training and validation.
'''),code('''from data_utils import demo_role_data, load_role_data
DEMO = True
if DEMO:
    frames, audit = demo_role_data()
else:
    frames, audit = load_role_data(ROOT / 'data/raw/roles', download=True)
print('SYNTHETIC TEACHING DEMO' if DEMO else 'OFFICIAL ROLE DATASET')
display(pd.DataFrame({name: {'documents': df.doc_index.nunique(), 'sentences': len(df)} for name, df in frames.items()}).T)
'''),md('''## 2. Inspect one training example
A sentence has text, a label, its judgment ID, and character offsets. We preserve legal wording, numbers and negation. Stop-word removal is a feature choice for some TF-IDF experiments, not a destructive edit to the source.
'''),code('''train = frames['train']
display(train[['doc_index', 'sentence_index', 'start', 'end', 'text', 'label']].head(8))
display(train.label.value_counts().rename('training_sentences').to_frame())
'''),md('''## 3. Understand the annotation policy
Some documents have more than one annotator. Merging their rows would duplicate sentences or attach contradictory labels. We use an explicitly adjudicated annotation when present; otherwise the first nonempty set, and record disagreements. We sort by offsets and reject duplicate spans, unknown labels and overlapping document splits.
The implementation is in `src/annotations.py`; the short public interface is `prepare_frames` in `src/data_utils.py`.
'''),code('''audit_summary = {name: {key: value for key, value in info.items() if key != 'multi_annotation_details'}
                 for name, info in audit.items() if isinstance(info, dict)}
display(pd.DataFrame(audit_summary).T)
print({key: value for key, value in audit.items() if 'duplicate_documents' in key})
'''),md('''## 4. Save the prepared splits
Notebook 02 reads these files. The demo and official data go to different folders, so an invented example cannot silently replace the real dataset.
'''),code('''from data_utils import save_frames
prepared = ROOT / 'data/processed' / ('demo' if DEMO else 'official')
save_frames(frames, audit, prepared)
print('Saved:', prepared)
'''),md('''## 5. Separate summarization data
The IN-Abs experiment used **1,000 train, 40 development and 100 official test cases**, with a 400-word summary budget and 120 sampled candidate sentences. The saved train/dev IDs are versioned. The official test has already been evaluated; do not repeatedly tune against it.
Download the official ZIP with `scripts/download_data.py inabs`; its MD5 is checked. Dataset URLs, licenses/access conditions and citations are in `data/README.md`.
'''),code('''split_ids = json.loads((ROOT / 'data/splits.json').read_text())
print({key: len(value) for key, value in split_ids.items() if key.endswith('_ids')})
'''),md('''### Explain it to your faculty
“I preserve sentence order and judgment boundaries, audit labels, and keep training and evaluation cases separate. OpenNyAI provides role labels; IN-Abs provides reference summaries. I never use a test summary to construct training labels.”

**Next:** 02 trains and compares models. **Ask yourself:** Why would splitting individual sentences randomly give an overly optimistic score?
''')]

two=[md('''# 02 · From TF-IDF to InLegalBERT and sentence selection
**Question:** What did each model add, and how was it trained?

The path is **TF-IDF baselines → word BiLSTM → context-aware InLegalBERT → document BiLSTM (with a CRF comparison) → sentence selector**.
The final app uses the existing verified encoder and heads. Heavy retraining is off by default. Short cells expose the decisions; reusable loops are in `src/train_models.py` and `src/train_selector.py`. Original experiment sources remain in `research/`.

For visible, step-by-step KNN, logistic regression, SVM and word-BiLSTM training code, open the optional [Stage 1 notebook](00_classical_ml_and_word_bilstm.ipynb). [Where the rest of the code lives](../docs/code_map.md).
'''),*setup,
code('''from data_utils import demo_role_data, read_frames
from train_models import train_baselines
from datetime import datetime, timezone
DEMO = True
RUN_TRAINING = False
frames = demo_role_data()[0] if DEMO else read_frames(ROOT / 'data/processed/official')
run = ROOT / 'outputs' / datetime.now(timezone.utc).strftime('training_%Y%m%d_%H%M%S')
train, dev = frames['train'], frames['dev']
'''),md('''## 1. Classical machine learning
TF-IDF converts a sentence into a sparse vector of word importance. KNN finds nearby examples; logistic regression and SVM learn decision boundaries. The majority classifier is the minimum baseline. Fit the vocabulary and IDF weights on training text only.
The demo runs quickly and illustrates `.fit()` and `.predict()`; its performance is **not** a real evaluation result.
'''),code('''baseline_scores = train_baselines(train, dev, run / 'baselines')
print('SYNTHETIC DEMO SCORES' if DEMO else 'NEW OFFICIAL DEV RUN')
display(baseline_scores)
'''),md('''## 2. Why a BiLSTM, and why then BERT?
The Stage 1 BiLSTM reads the words of one sentence in both directions. It learns order, but starts with randomly initialized word embeddings.
InLegalBERT starts with a language model pretrained on Indian legal text. Fine-tuning teaches it our 13 roles. Stage 2.2 compared **target-only 256 tokens** with **target plus neighboring sentences within 256 tokens**. The target gets at most 192 tokens in the context experiment. Neighbors must come from the same judgment.
'''),code('''from stage44_model import SentenceBiLSTM, SentenceSelector
print('Document BiLSTM:', SentenceBiLSTM())
print('Sentence selector without role features:', SentenceSelector(with_roles=False))
'''),md('''## 3. Fine-tune the encoder (optional GPU run)
For each batch: predict roles → calculate cross-entropy loss → backpropagate → update weights. Four batches accumulate gradients before an optimizer step. We save the epoch with the best **development macro F1**, not the highest training accuracy.
Three epochs, learning rate 2e-5, effective batch size 32. The two experiments use the same data and principal settings.
These cells run only when `DEMO=False` and `RUN_TRAINING=True`; they can take hours. New outputs are separate from the deployed model.
'''),code('''from train_models import train_encoder
if RUN_TRAINING and not DEMO:
    target_checkpoint = train_encoder(train, dev, run / 'target256', mode='target256')
    context_checkpoint = train_encoder(train, dev, run / 'context256', mode='context256')
else:
    print('Encoder training skipped. Recorded results are in Notebook 03.')
'''),md('''## 4. Read the whole judgment as a sequence
The second BiLSTM is different from the Stage 1 word model: it reads **768-dimensional sentence vectors**, in judgment order. It predicts one role per sentence. Padding is masked from the loss.
The optional CRF learns which role sequences are plausible and uses Viterbi decoding. It was a controlled experiment, not a guaranteed improvement. The deployed role head uses softmax.
'''),code('''from train_models import document_embeddings, train_role_head
if RUN_TRAINING and not DEMO:
    train_sequences = document_embeddings(train, context_checkpoint)
    dev_sequences = document_embeddings(dev, context_checkpoint)
    for use_crf in [False, True]:
        train_role_head(train_sequences, dev_sequences, run / ('crf' if use_crf else 'softmax'),
                        context_checkpoint, use_crf=use_crf)
else:
    print('Document-head training skipped.')
'''),md('''## 5. How does the model learn what to include?
A role label is not an importance label. For IN-Abs training judgments, we create **weak sentence labels** using the training reference summary. One teacher ranks TF-IDF similarity; another greedily adds new reference unigram/bigram coverage. Both respect a 400-word budget.
These are heuristic labels, not expert sentence annotations. At inference the reference summary is unavailable and is never a model input.
'''),code('''from selection_training import tfidf_budgeted_labels, greedy_oracle_labels
sentences = ['The claimant challenged the payment rejection before the court.',
             'The court found the claim was filed within time.',
             'The appeal is allowed and payment is directed within six weeks.']
training_reference = 'The appeal is allowed. Payment must be made within six weeks.'
display(pd.DataFrame({'sentence': sentences,
    'TF-IDF teacher': tfidf_budgeted_labels(sentences, training_reference, budget=24),
    'Coverage teacher': greedy_oracle_labels(sentences, training_reference, budget=24)}))
'''),md('''## 6. Train four matched selectors (optional)
Two teachers × role features on/off gives four conditions. Every selector uses a sentence embedding plus normalized position; the role-aware version also receives 13 predicted role probabilities. A small neural network produces an importance score. Binary cross-entropy trains it; development ROUGE-L selects the checkpoint.
This stage deliberately uses the original frozen encoder/head through `load_deployed_model`. Newly retrained encoders/heads require a new feature pass and provenance bundle; they must not be mixed with old selectors. New research checkpoints do not silently replace the app's verified checkpoints.
'''),code('''from project_api import load_deployed_model
from train_selector import prepare_selector_cases, train_selectors
if RUN_TRAINING and not DEMO:
    frozen_model = load_deployed_model(ROOT)
    cases = prepare_selector_cases(ROOT / 'data/raw/dataset.zip', ROOT / 'data/splits.json', frozen_model)
    history = train_selectors(cases, run / 'selectors')
else:
    print('Selector training skipped. No weights changed during this demonstration.')
'''),md('''### Explain it to your faculty
“The encoder represents legal sentences; the document BiLSTM predicts their roles; a separately trained selector ranks importance. I compared role-aware selection with a no-role control instead of assuming roles would help.”

Optional CRF, weighted-loss experiments and the earlier word-level BiLSTM are documented in the source archive and methodology. Consolidated training entrypoints retain the main architectures/settings but have not been rerun to reproduce all recorded numbers. Notebook 03 labels the recorded runs explicitly.
''')]

three=[md('''# 03 · Evaluation: what improved, and what still fails?
**Question:** What evidence supports the project, and what does it not establish?

We report three different things separately: role classification, summary overlap, and source/decision review. The tables below are **recorded results from previous runs**, not scores produced by this notebook. No training or test-set tuning happens here.
'''),*setup,
md('''## 1. Role classification
Accuracy counts all correct labels. **Macro F1** gives every role equal weight, including rare ones. Weighted F1 gives common roles more influence. These numbers are on the development split; they are not unseen-document legal accuracy.
'''),code('''roles = pd.read_csv(ROOT / 'evaluation/role_results.csv')
display(roles.round(4))
ax = roles.plot.barh(x='model', y='macro_f1', figsize=(9, 5), legend=False, title='Recorded DEV macro F1')
ax.set_xlabel('Macro F1 (0–1)')
'''),md('''## 2. Summarization on the original frozen protocol
ROUGE measures overlap with reference wording; it does not verify that the selected court decision is current or correct. The recorded Stage 3.6 test used 100 official cases, 120 sampled candidates and a 400-word budget. It had already been run once before this consolidation.
No model wins every metric. Report the entire comparison, not just the best number. Role features did not consistently improve the results.
'''),code('''results = pd.read_csv(ROOT / 'evaluation/evaluation_results.csv')
display(results.round(4))
results.plot.bar(x='method', y=['rouge1_f1', 'rouge2_f1', 'rougeL_f1'], figsize=(10, 4),
                 title='Recorded Stage 3.6 official test (100 cases)')
'''),md('''## 3. Earlier 4.6 pipeline: measured automatic scores
This table is from an actual run of the frozen current pipeline on all 100 official IN-Abs test judgments. The primary setting is `pipeline_800`; `lead_800` is the opening-sentence baseline. The 400-word rows are a secondary comparison.
Scores are ROUGE F1 points on a 0–100 scale, not percentages of legally correct summaries. The test set was used before, so this is a reused benchmark. The code and weights were not tuned during this run. Read `docs/current_summary_benchmark.md` for lengths, failures, paired differences and limitations.
'''),code('''measured = json.loads((ROOT / 'evaluation/current_summary_benchmark.json').read_text())
rows = []
for method, values in measured['methods'].items():
    interval = values['rougeL_f1']
    rows.append({'method': method, 'ROUGE-1': 100 * values['rouge1_f1']['mean'],
                 'ROUGE-2': 100 * values['rouge2_f1']['mean'], 'ROUGE-L': 100 * interval['mean'],
                 'L lower 95%': 100 * interval['ci95_low'], 'L upper 95%': 100 * interval['ci95_high']})
display(pd.DataFrame(rows).round(2))
print('Cases:', measured['case_count'], '| Legal accuracy measured:', measured['automatic_legal_accuracy'])
'''),md('''### 3b. Version 4.7: compare the upgrade on the same reserved cases
This is a different, 100-case sample from unused IN-Abs training pairs, excluding the known 1,040 selector train/dev IDs and exact source duplicates. The candidate was tuned on the old 40-case development set and frozen before this evaluation. Read `docs/coverage_upgrade.md` for the exposure audit and limitations.

The code still uses the same trained weights. It fixes quotation handling, recognizes more disposition forms, avoids copying huge paragraphs as decisions and spends unused summary space on ranked evidence. Compare the old and upgraded rows **within this table**, not across different datasets. More populated decision sections do not prove all decisions are correctly attributed.
'''),code('''upgrade = json.loads((ROOT / 'evaluation/coverage_upgrade_reserved.json').read_text())
rows = []
for method, values in upgrade['methods'].items():
    rows.append({'method': method,
                 'ROUGE-1': 100 * values['rouge1_f1']['mean'],
                 'ROUGE-2': 100 * values['rouge2_f1']['mean'],
                 'ROUGE-L': 100 * values['rougeL_f1']['mean'],
                 'mean words': values['mean_words'],
                 'empty decisions': values['empty_decision'] if not method.startswith('lead') else None})
display(pd.DataFrame(rows).round(2))
print('Reserved cases:', upgrade['cases'], '| Legal accuracy measured:', upgrade['automatic_legal_accuracy'])
print('Paired 800-word ROUGE-L change:', upgrade['paired']['800_minus_baseline'])
'''),md('''## 4. Why historical scores and source checks are separate
The app now scores the full extracted judgment in overlapping windows and builds structured sections with evidence checks. That changes the inference protocol. The Stage 3.6 numbers describe the earlier fixed-budget experiment, not the current app.
All extracted sentences being scored means coverage of extracted candidates; it does not prove flawless PDF extraction or complete summaries.
'''),code('''review = json.loads((ROOT / 'evaluation/current_pipeline_review.json').read_text())
display(pd.DataFrame([{'file': x['file'], 'scored': x['scored'], 'extracted': x['extracted_sentences'],
    'exact_evidence': x['exact_evidence'], 'critical_decision_failure': x['critical_decision_failure']}
    for x in review['documents']]))
print('Human legal accuracy measured:', review['human_accuracy_measured'])
'''),md('''## 5. Exact evidence is necessary, not sufficient
One separate case put an earlier High Court ruling in the current decision section. Another missed the final decision section. Both can still contain copied source text. **Two of three separate cases had serious decision-section failures.** The comparison was performed by the assistant, not independent legal annotators; training overlap is unknown.
This table records the original failures. Those cases later became development examples for the 4.6 decision fixes; their targeted checks now pass. This does not establish complete legal correctness. See `docs/decision_upgrade.md`.
'''),code('''for item in review['documents']:
    print(item['file'])
    print(item['finding'])
    print()
'''),md('''## 6. A simple ROUGE demonstration
The following invented pair illustrates lexical overlap only. A high score can coexist with a changed condition or outcome.
'''),code('''from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
reference = 'The appeal is allowed and payment is due within six weeks.'
prediction = 'The appeal is dismissed and payment is due within six weeks.'
display(pd.DataFrame({name: value._asdict() for name, value in scorer.score(reference, prediction).items()}).T)
print('One changed word reverses the outcome. ROUGE alone cannot validate this summary.')
'''),md('''### What to say, and what not to claim
“The recorded role-classification dev macro F1 improved from about 0.39 for logistic regression to 0.65 for the document BiLSTM. The earlier 100-case summarization test reached roughly 0.52 ROUGE-1 F1. The current pipeline has its own measured ROUGE results above, with a baseline and confidence intervals. Neither result is legal accuracy, so I present the app as a research assistant.”

Do not say “83% summary accuracy,” “error-free,” “best model,” or “production ready.” 83% is approximately a role-classification development accuracy, a different task.
''')]

four=[md('''# 04 · Final pipeline and explainable demonstration
**Question:** How does a new judgment become a reviewable brief?

**PDF/TXT → source sentences → trained scores and roles → evidence-grounded sections → audit → human review.**
The existing trained weights and checkpoint checks are preserved. Version 4.7 improves quote handling, decision recognition and use of the summary budget. See `docs/coverage_upgrade.md` and Notebook 03 for the same-corpus reserved comparison and remaining limits. Reference overlap is not legal accuracy.
'''),*setup,
md('''## 1. Choose the demonstration mode
The invented TXT example runs without downloads. For a real trained-model demo, run `scripts/download_models.py` once, set `USE_TRAINED_MODEL=True`, and provide a judgment path. The supplied model remains your actual fine-tuned encoder and trained heads; we do not substitute a base BERT model.
'''),code('''from project_api import load_deployed_model, summarize_document, save_summary
USE_TRAINED_MODEL = False
INPUT = ROOT / 'examples/demo_judgment.txt'
BUDGET = 800
model = load_deployed_model(ROOT) if USE_TRAINED_MODEL else None
print('TRAINED MODEL' if model else 'EVIDENCE-ONLY TEACHING DEMO: no trained ranking')
'''),md('''## 2. Inspect what was extracted
The PDF reader preserves page references and character spans. Quotations, earlier rulings and party submissions require attribution checks. OCR is optional and needs Tesseract; start with a searchable English PDF for your presentation.
'''),code('''from stage44_source import read_document
document = read_document(INPUT, ocr='off')
display(pd.DataFrame(document.sentences)[['id', 'page', 'sentence']].head(10))
print('Extracted sentences:', len(document.sentences))
'''),md('''## 3. Why long judgments no longer stop at sentence 120
The trained selector originally used 120 sampled candidates. The current inference engine applies it to overlapping windows of 120 sentences, with overlap 30, then averages scores where windows overlap. This covers all extracted candidates, including the ending. It is a new inference regime and must be evaluated separately.
'''),code('''from stage44_windows import window_ranges
print('Windows for a hypothetical 350-sentence judgment:')
print(list(window_ranges(350)))
'''),md('''## 4. Produce the brief
The model ranks candidates; source and attribution rules organize background, issues, reasoning and directions. Complete operative passages can exceed the requested word target. Exact-span checks reject modified source evidence. A missing section is shown explicitly rather than invented.

After the core sections are selected, remaining space is filled using learned importance and a penalty for repeated content. Additional sentences must fit the remaining allowance. Model-only reasoning candidates are marked for attribution review; a high score cannot turn a known letter or submission into court reasoning.
'''),code('''result = summarize_document(INPUT, model=model, budget=BUDGET, ocr='off')
display(Markdown(result['markdown']))
'''),md('''## 5. Inspect the audit before trusting the result
Exact text matching checks copying. It cannot prove attribution, completeness or legal correctness. Compare the final pages manually, especially the outcome, amounts, recipients, deadlines, exceptions and negation.
'''),code('''display(pd.DataFrame([{'mode': result['model']['mode'],
    'exact_source_check': result['audit']['exact_source_check_passed'],
    'missing_sections': ', '.join(result['audit']['missing_sections']),
    'flags': ', '.join(result['audit']['flags'])}]))
saved = save_summary(result, ROOT / 'outputs/presentation_demo')
print('Saved summary and evidence:', saved)
'''),md('''## 6. Use the browser app
On a computer with the dependencies and model installed, run `python src/stage44_app.py` from the repository. Open `http://127.0.0.1:7861` **on that same computer**. Colab's remote localhost is not your laptop's localhost.
The app shows source pages, exact highlights and separately saved reviewer drafts. It is a local research preview; the Colab batch notebook is the supported cloud workflow.

### A two-minute live demo
1. Upload a short searchable judgment.
2. Generate the brief and open an evidence reference.
3. Jump to the final pages and check the decision.
4. Explain a warning, correct the reviewer draft and export it.

**Next research step:** obtain independent paragraph/decision annotations, improve attribution and calibrated abstention, and evaluate on a new untouched set. Do not promise that the existing three failure cases prove performance after fixing them.
''')]

for name,cells in [('01_data_preprocessing',one),('02_model_training',two),('03_model_evaluation',three),('04_final_pipeline',four)]:
 for i,c in enumerate(cells):c['id']=f'{name[:2]}-{i:03}'
 nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':name+'.ipynb'}},nbformat=4,nbformat_minor=5)
 (ROOT/'notebooks'/(name+'.ipynb')).write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf8')
 print(name,len(cells),'cells')
