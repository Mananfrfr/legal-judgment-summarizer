"""Build an optional Stage 1 notebook with the training logic visible inline."""
from pathlib import Path
import copy,json,textwrap
ROOT=Path(__file__).resolve().parents[1]

def md(text):
    return dict(cell_type='markdown',metadata={},source=textwrap.dedent(text).strip().splitlines(True))

def code(text):
    return dict(cell_type='code',metadata={},source=('%%project\n'+textwrap.dedent(text).strip()+'\n').splitlines(True),execution_count=None,outputs=[])

# Reuse only the common setup cells; the ML logic below is deliberately inline.
base=json.loads((ROOT/'notebooks/01_data_preprocessing.ipynb').read_text(encoding='utf8'))
setup=copy.deepcopy(base['cells'][1:4])
cells=[md('''# 00 · Stage 1: classical ML and a word-level BiLSTM
**Optional detailed walkthrough before Notebook 02.** The four main notebooks still cover the complete project.

Here you can see the actual `.fit()`, `.predict()` and neural training loop rather than a single helper call.
We compare **majority baseline → TF-IDF + KNN / logistic regression / linear SVM → word BiLSTM**.
Each model predicts a sentence's rhetorical role; none of these models alone generates a summary.

The default uses invented data and runs quickly. Its scores are software demonstrations, **not project accuracy**.
Recorded Stage 1 results are shown separately at the end. This readable rerun uses the audited shared annotation policy; it does not claim exact reproduction of the historical scores.
'''),*setup,
md('''## 1. Choose the data and keep experiments separate
`DEMO=True` uses invented sentences. To use official data, first follow `data/README.md`, then set `DEMO=False`.
All models see the same TRAIN and DEV judgments. The official TEST split is not used to choose models or settings.
The data loader audits annotation spans and rejects judgment overlap between splits.

The optional BiLSTM training is off for the live presentation. Every run gets a new output folder so it cannot overwrite the deployed model.
'''),code('''from data_utils import demo_role_data, load_role_data
from annotations import LABELS, LABEL2ID, ID2LABEL
from datetime import datetime, timezone
import numpy as np

DEMO = True
RUN_WORD_BILSTM = False
WORD_EPOCHS = 15
SEED = 42

frames, audit = demo_role_data() if DEMO else load_role_data(ROOT / 'data/raw/roles', download=True)
train, dev = frames['train'], frames['dev']
run = ROOT / 'outputs' / datetime.now(timezone.utc).strftime('stage1_%Y%m%d_%H%M%S_%f')
run.mkdir(parents=True, exist_ok=False)
print('SYNTHETIC DEMO — not project results' if DEMO else 'NEW OFFICIAL-DATA DEVELOPMENT RUN')
display(pd.DataFrame({'train': [len(train), train.doc_index.nunique()],
                      'dev': [len(dev), dev.doc_index.nunique()]}, index=['sentences', 'judgments']))
display(train[['text', 'label']].head())
'''),md('''## 2. Turn words into TF-IDF features
TF measures how often a word appears in a sentence. IDF gives less weight to words that occur in many training sentences.
The result is a sparse numerical vector. We use at most 5,000 vocabulary features, matching the Stage 1 setting.

**Why `fit_transform` on TRAIN but only `transform` on DEV?** Learning the vocabulary or IDF from DEV would leak evaluation information into training.
We do not strip negation or edit the original judgment text.
'''),code('''from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train.text)
X_dev = vectorizer.transform(dev.text)
print('TRAIN matrix:', X_train.shape, '| DEV matrix:', X_dev.shape)
features = vectorizer.get_feature_names_out()
first_sentence = pd.Series(X_train[0].toarray().ravel(), index=features)
display(first_sentence[first_sentence > 0].sort_values(ascending=False).head(10).rename('TF-IDF weight'))
'''),md('''## 3. Define the four classical models
| Model | Simple explanation | Why include it? |
|---|---|---|
| Majority | Always predicts the most frequent training role | Checks whether a model beats a trivial answer |
| KNN (k=5) | Finds five nearby training sentence vectors and votes | A simple similarity baseline |
| Logistic regression | Learns a weighted score for each role, then probabilities | A strong, interpretable linear baseline |
| Linear SVM | Learns separating boundaries with a wide margin | Another strong baseline for sparse text vectors |

`class_weight='balanced'` makes rare roles count more during logistic regression/SVM training.
The settings follow the historical Stage 1 comparison. KNN can become slow on the full corpus because prediction compares against stored training examples.
'''),code('''from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

models = {
    'majority': DummyClassifier(strategy='most_frequent'),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'logreg': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED),
    'svm': LinearSVC(max_iter=5000, class_weight='balanced', random_state=SEED),
}
'''),md('''## 4. Train, predict, measure and save
`.fit()` learns from TRAIN. `.predict()` assigns roles to DEV sentences the model did not train on.
Accuracy is the fraction of correct labels. **Macro F1** gives all 13 roles equal importance; weighted F1 gives frequent roles more influence.
We compare on DEV, keeping TEST out of this loop.
'''),code('''from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
import joblib

def measure(truth, predicted):
    return {
        'accuracy': accuracy_score(truth, predicted),
        'macro_f1': f1_score(truth, predicted, labels=LABELS, average='macro', zero_division=0),
        'weighted_f1': f1_score(truth, predicted, labels=LABELS, average='weighted', zero_division=0),
    }

scores, predictions = [], {}
for name, model in models.items():
    model.fit(X_train, train.label)
    predictions[name] = model.predict(X_dev)
    scores.append({'model': name, 'split': 'dev', **measure(dev.label, predictions[name])})
    joblib.dump({'vectorizer': vectorizer, 'model': model}, run / f'{name}.joblib')

current_scores = pd.DataFrame(scores)
current_scores.to_csv(run / 'classical_metrics.csv', index=False)
print('CURRENT SYNTHETIC DEMO' if DEMO else 'CURRENT OFFICIAL DEV RUN')
display(current_scores.sort_values('macro_f1', ascending=False))
'''),md('''## 5. Inspect mistakes instead of reporting one number
Choose the strongest **classical** model on DEV macro F1 for inspection. A confusion matrix shows which roles get mixed up.
Even a perfect score on this repetitive invented dataset says nothing about unseen real judgments.
On official data, inspect mistakes and rare-role recall before deciding what to improve.
'''),code('''import matplotlib.pyplot as plt

best_name = current_scores.sort_values('macro_f1', ascending=False).iloc[0]['model']
report = classification_report(dev.label, predictions[best_name], labels=LABELS, output_dict=True, zero_division=0)
display(pd.DataFrame(report).T)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(dev.label, predictions[best_name], labels=LABELS,
    xticks_rotation=90, ax=ax, colorbar=False)
ax.set_title(('Synthetic demo' if DEMO else 'Official DEV') + ': ' + best_name)
fig.tight_layout()
fig.savefig(run / 'confusion_matrix.png', dpi=150)
plt.show()
errors = dev[['text', 'label']].copy()
errors['predicted'] = predictions[best_name]
display(errors[errors.label != errors.predicted].head(10))
'''),md('''## 6. Predict a new sentence and reload a saved model
The vocabulary stays fixed: unseen words are ignored by TF-IDF. This illustrates why sparse word features can struggle with new wording.
The predicted role below is a model guess, not a validated label or legal advice. Only load model files you created or trust.
'''),code('''sentence = 'The appeal is allowed and the impugned order is set aside.'
saved = joblib.load(run / f'{best_name}.joblib')
guess = saved['model'].predict(saved['vectorizer'].transform([sentence]))[0]
print('Sentence:', sentence)
print('Predicted role:', guess)
assert np.array_equal(saved['model'].predict(saved['vectorizer'].transform(dev.text)), predictions[best_name])
'''),md('''## 7. Why try a word-level BiLSTM?
TF-IDF mostly loses word order. A BiLSTM reads a sentence forwards and backwards and learns word embeddings from our training data.

**Flow:** words → integer IDs → 100-dimensional embeddings → bidirectional LSTM (128 units per direction) → 13 role scores.
The original settings use 50 words per sentence, a 10,000-word vocabulary cap, dropout 0.3, Adam at 0.001, batch size 64 and up to 15 epochs.
`<PAD>` fills short sentences; `<UNK>` represents unseen words. Packing tells the LSTM to ignore padding.

This is the **Stage 1 word model**, not the later document BiLSTM that reads BERT sentence vectors. Its weights remain separate from the app.
'''),code('''import random
from collections import Counter
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = dict(max_words=50, vocab_size=10000, embed_dim=100, hidden_dim=128,
              batch_size=64, lr=0.001, patience=3, epochs=WORD_EPOCHS, seed=SEED)

class WordBiLSTM(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config['embed_dim'], padding_idx=0)
        self.bilstm = nn.LSTM(config['embed_dim'], config['hidden_dim'], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(2 * config['hidden_dim'], len(LABELS))

    def forward(self, words, lengths):
        packed = pack_padded_sequence(self.embedding(words), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.bilstm(packed)
        both_directions = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(self.dropout(both_directions))
'''),md('''## 8. Prepare the neural inputs (optional)
Build the vocabulary from TRAIN only. Unknown DEV words become `<UNK>`.
The loaders group sentences into batches; each batch carries true lengths and labels.
Set `RUN_WORD_BILSTM=True` above and rerun from that cell to enable this path. On the demo, one epoch is enough to check the code; official training can take substantially longer.
'''),code('''if RUN_WORD_BILSTM:
    counts = Counter(word for text in train.text for word in text.lower().split())
    vocabulary = ['<PAD>', '<UNK>'] + [word for word, _ in counts.most_common(config['vocab_size'] - 2)
                                      if word not in {'<PAD>', '<UNK>'}]
    word2idx = {word: i for i, word in enumerate(vocabulary)}

    def make_dataset(frame):
        encoded = [[word2idx.get(word, 1) for word in text.lower().split()[:config['max_words']]] or [1]
                   for text in frame.text]
        lengths = torch.tensor([len(words) for words in encoded])
        padded = torch.tensor([words + [0] * (config['max_words'] - len(words)) for words in encoded])
        labels = torch.tensor([LABEL2ID[label] for label in frame.label])
        return TensorDataset(padded, lengths, labels)

    train_loader = DataLoader(make_dataset(train), batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    dev_loader = DataLoader(make_dataset(dev), batch_size=64, shuffle=False)
    word_model = WordBiLSTM(len(vocabulary)).to(device)
    print(word_model)
else:
    print('Word BiLSTM training is disabled; the architecture above remains visible.')
'''),md('''## 9. Train and select the best development checkpoint (optional)
For each batch: predict → compute cross-entropy → backpropagate → clip large gradients → update weights.
Rare training roles receive greater loss weight. After each epoch, evaluate DEV macro F1.
Save the best checkpoint and stop after three epochs without improvement. Never select a checkpoint using TEST.
'''),code('''if RUN_WORD_BILSTM:
    counts = train.label.value_counts().reindex(LABELS, fill_value=0).to_numpy()
    present = counts > 0
    weights = np.zeros(len(LABELS), dtype=np.float32)
    weights[present] = len(train) / (present.sum() * counts[present])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.Adam(word_model.parameters(), lr=config['lr'])

    @torch.no_grad()
    def predict_words():
        word_model.eval()
        result = []
        for words, lengths, labels in dev_loader:
            ids = word_model(words.to(device), lengths).argmax(1).cpu().tolist()
            result.extend(ID2LABEL[i] for i in ids)
        return result

    best, stale, history = -1.0, 0, []
    for epoch in range(1, config['epochs'] + 1):
        word_model.train()
        losses = []
        for words, lengths, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(word_model(words.to(device), lengths), labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(word_model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        metrics = measure(dev.label, predict_words())
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), **metrics})
        print(history[-1])
        if metrics['macro_f1'] > best:
            best, stale = metrics['macro_f1'], 0
            torch.save(word_model.state_dict(), run / 'word_bilstm.pt')
        else:
            stale += 1
        if stale >= config['patience']:
            break

    word_model.load_state_dict(torch.load(run / 'word_bilstm.pt', map_location=device, weights_only=True))
    pd.DataFrame(history).to_csv(run / 'word_training_history.csv', index=False)
    (run / 'word_config.json').write_text(json.dumps(config, indent=2))
    (run / 'word_vocabulary.json').write_text(json.dumps(word2idx))
    (run / 'word_labels.json').write_text(json.dumps(LABEL2ID))
    word_scores = pd.DataFrame([{'model': 'word_bilstm', 'split': 'dev', **measure(dev.label, predict_words())}])
    word_scores.to_csv(run / 'word_metrics.csv', index=False)
    display(word_scores)
'''),md('''## 10. What did the original Stage 1 experiments actually show?
The table below comes from the **saved historical Colab output**, rounded to six decimals. It is not the result of today's demo or optional rerun.
Its source is recorded in `evaluation/recorded/provenance.json` and the original code is retained under `research/`.
The word BiLSTM improved the recorded role scores, but still started with embeddings learned from limited task data. That motivated testing pretrained InLegalBERT next.
'''),code('''recorded = pd.read_csv(ROOT / 'evaluation/recorded/stage1_dev.csv')
print('RECORDED HISTORICAL STAGE 1 RESULTS — NOT THIS RUN')
display(recorded)
print("This run's separate artifacts:", run)
'''),md('''### Explain this in one minute
“I first treated sentence-role prediction as a classification task. TF-IDF converts text into features, and I compared majority, KNN, logistic regression and SVM using the same development split. Then I trained a word-level BiLSTM to model word order. I fitted vocabularies only on training data and used macro F1 because roles are imbalanced. InLegalBERT was the next step because it already contains legal-language knowledge.”

**Next:** Notebook 02 for InLegalBERT, the document BiLSTM and sentence selection; Notebook 03 for all recorded comparisons.
See [where the code lives](../docs/code_map.md). The initial word model and the final document model are different experiments, not interchangeable checkpoints.
''')]

name='00_classical_ml_and_word_bilstm'
for i,cell in enumerate(cells):cell['id']=f'00-{i:03}'
nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':name+'.ipynb'}},nbformat=4,nbformat_minor=5)
(ROOT/'notebooks'/(name+'.ipynb')).write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf8')
print(name,len(cells),'cells')
