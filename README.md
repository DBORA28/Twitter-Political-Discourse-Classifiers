# Twitter Political Discourse Classifiers

Two-stage pipeline for analyzing political discourse on Twitter, built on a Hinglish/code-mixed dataset with no ground-truth labels:

1. **Political vs. Non-Political** — analysing user tweets and separate them out as poltical tweet or non-poltical?
Any tweet will be considered poltical if there are poltical mentions or hashtags or any suitable poltical refernce inside the tweets
2. **Politically Inclined vs. Uninclined** — analysing user tweets and infer whether does this tweet express an opinion/stance, or is it neutral/factual?

Both stages use a layered, cascading approach rather than a single model — each layer catches what the previous one missed, since no single signal (keyword matching, sentiment, entity recognition) is reliable alone on informal, code-mixed social media text.

---

## Notebook 1 — Political vs. Non-Political Classification

**File:** pol/non-pol.ipynb

### Goal
Label every tweet as political (1) or non-political (0) without any pre-existing ground truth, using a cascade of signals — each stage only processes the tweets the previous stage couldn't confidently label.

### Pipeline

```
Raw tweets
   │
   ▼
Preprocessing: dedupe, drop empty tweets, clean text (strip URLs, mentions,
hashtags→words, non-ASCII, punctuation; remove English + Hindi/Hinglish stopwords)
   │
   ▼
STAGE A — Mention & Hashtag Matching
  • Extract every @mention and #hashtag per tweet
  • Build frequency tables (top 200-300 of each)
  • Manually curate a non-political exclusion set (sports, entertainment,
    crypto, generic platforms, celebrity/entertainment hashtags, etc.)
  • tweet_label = 1 if tweet contains a known political mention OR hashtag
   │
   ▼ (tweets with no mention/hashtag signal)
STAGE B — TF-IDF Political Vocabulary Scoring
  • Extract top TF-IDF words corpus-wide, filter via stopwords + a curated
    DROP_FROM_VOCAB list (generic filler words with no topical meaning)
  • Restrict a second TF-IDF pass to only the surviving "political vocabulary"
  • is_political = 1 if any political-vocab word scores > 0 in the tweet body
   │
   ▼ (tweets still unlabeled)
STAGE C — Named Entity Recognition (spaCy)
  • Detect PERSON / ORG / GPE / NORP entities in the remaining tweets
  • Cross-check entities against a combined knowledge base
    (political mentions ∪ political hashtags ∪ TF-IDF political vocabulary)
  • ner_label = 1 if a detected entity matches a known political reference
   │
   ▼ (tweets still unlabeled)
STAGE D — Topic Modeling Fallback (NMF / LDA)
  • Unsupervised topic extraction (5-12 topics) on whatever remains
  • Each topic manually inspected and mapped to political (1) / non-political (0)
    based on its top keywords
```

### Key design decisions & why

- **Mention/hashtag matching first**: cheapest, most precise signal — if a tweet explicitly tags a known politician or political hashtag, that's strong, low-ambiguity evidence. Handles the majority of clearly political tweets with minimal false positives.
- **TF-IDF vocabulary as a fallback, not the primary signal**: frequency-based filtering (stopwords, document-frequency thresholds) alone cannot separate "frequent and political" from "frequent and non-political" — words like `watched` or `dog` pass the exact same statistical tests as `modi`. The political-vocabulary list had to be manually curated on top of frequency filtering, not derived from frequency alone.
- **NER only runs on the leftover subset**: spaCy's NER is expensive relative to a keyword lookup, so it's deliberately scoped to only the tweets that stages A and B couldn't already resolve — this keeps the pipeline fast without sacrificing coverage.
- **Topic modeling as the last resort**: for tweets with no mentions, no hashtags, no political vocabulary hits, and no recognized entities, there's no shortcut left — unsupervised topic discovery plus manual topic labeling is the only remaining option.
- **Manual exclusion lists needed constant iteration**: the non-political mention/hashtag sets were refined repeatedly as new noise categories surfaced (cricket, crypto, health/COVID hashtags, celebrity birthday tags, generic lifestyle content, foreign geopolitical hashtags) — this is a running list, not a one-time filter.

---

## Notebook 2 — Politically Inclined vs. Uninclined Classification

**File:** `notebook7aefd5c5fb (1).ipynb`

### Goal
Within tweets already labeled political (via Notebook 1's `is_political` signal), distinguish **inclined** (expresses a political opinion/stance) from **uninclined** (politically-relevant but neutral/factual, or not political at all).

### Two approaches built and compared

**Approach A — K-Means Clustering**: combine the filtered political-vocabulary TF-IDF matrix with VADER's `compound` sentiment score (weighted ×5), cluster into 2 groups, and label whichever cluster has the higher average `|compound|` as "inclined."

**Approach B — Rule-Based Classification**: no clustering — a direct rule: `is_political == 1 AND abs(compound) > threshold → inclined`, otherwise uninclined. Threshold (0.4) chosen by sweeping several candidate values and checking the resulting class balance.

Both approaches were then used to train a **LinearSVC** classifier (tuned via `GridSearchCV` over the regularization parameter `C`, scored on macro-F1) on TF-IDF + VADER compound features, so the labels generated by clustering/rules could be validated against a trainable model — with Approach A's high-confidence predictions (SVM probability ≥ 0.75, via `CalibratedClassifierCV`) filtered out as clean training data for a downstream LSTM.

### The critical finding — and why it matters more than the accuracy numbers

Initial accuracy for both approaches looked strong (89–94%), but diagnostic testing on hand-picked tweets exposed a structural flaw:

- `"I love watching cricket matches on weekends"` → predicted **politically_inclined** (false positive)
- `"Politicians are all corrupt and should be voted out"` → predicted **politically_uninclined** (false negative)

**Root cause**: both approaches used `abs(VADER compound) > threshold` as a proxy for "has a political stance." But VADER measures general emotional sentiment, not political stance — calm, clearly partisan statements ("I strongly support BJP's economic policies") score low on VADER and get missed, while emotionally-charged but non-political tweets score high and get falsely flagged. **Sentiment intensity and political stance are not the same thing**, and the 89–94% accuracy was inflated because the SVM was learning to reproduce its own label-generating rule (VADER-threshold), not genuine political stance — a form of circularity, not real predictive skill.

### The fix

1. **Added an explicit stance lexicon** (`STANCE_SUPPORT`, `STANCE_OPPOSE` — words like "support," "endorse," "condemn," "corrupt," "boycott") as an independent `stance_hit` signal, separate from VADER.
2. **New label rule**: `is_political AND (stance_hit OR abs(compound) > threshold)` — stance detection is additive (an OR), so it recovers calm-but-partisan tweets without losing anything the old VADER-only rule already caught.
3. **Verified the narrower 663-word "political vocabulary" TF-IDF couldn't carry most stance words** (they'd been filtered out as "generic sentiment" words during Stage B of Notebook 1) — switched the classifier's input features to the **full 5,000-word TF-IDF** plus `compound` plus `stance_hit`, so stance language is actually visible to the model.
4. **Result**: held-out accuracy dropped to 74.03% (from the inflated 89–94%) — expected and correct, since this is now measuring a genuinely harder, more honest task. The four disputed test tweets, including the exact false-positive/false-negative cases above, were re-verified end-to-end after the fix.

### Why this is worth documenting, not hiding

A lower, honest accuracy number after fixing a circular evaluation is a *better* result than a higher, inflated one — the initial 89–94% was measuring the model's ability to mimic its own labeling rule, not its ability to detect real political stance. This distinction (and the diagnostic process used to catch it) is the most important part of this notebook.

---

## Requirements

```bash
pip install pandas numpy scikit-learn nltk spacy vaderSentiment matplotlib gensim
python -m spacy download en_core_web_sm
```

## Notes & Limitations

- Both notebooks operate on Hinglish/code-mixed tweets — cleaning includes a manually curated Hindi/Hinglish stopword list alongside NLTK's English stopwords.
- Every exclusion/inclusion list (non-political mentions, non-political hashtags, political vocabulary, stance lexicon) was built and refined manually/iteratively — none of it is derived automatically from labeled ground truth, since none exists for this dataset.
- Accuracy figures throughout should be read in context: without genuine ground-truth labels, all reported metrics measure agreement with the pipeline's own heuristic rules, not real-world classification accuracy. The Notebook 2 diagnostic section demonstrates why this distinction matters in practice.
