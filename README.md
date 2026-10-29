# Machine Credibility: How News Readers Evaluate AI-Generated Content

Extends and restructures the misinformation-detection pipeline into a credibility / readability / engagement /
explainability analysis tool for AI-generated news content.

## What's reused from the original repo vs. new

| Component | Status |
|---|---|
| DeBERTa check-worthiness model | reused (you must copy the model folder in manually - see below) |
| RoBERTa AI-text detector | reused (same HF model ID) |
| Tavily search + MiniLM relevance ranking | rewritten (`core/evidence.py`), same design |
| Groq NLI verdict + risk formula | rewritten (`core/verdict.py`), same formula and calibrated weights |
| Readability + coherence metrics | rewritten (`core/metrics.py`) |
| Composite credibility score | **new** (`analysis/credibility.py`) |
| Engagement / sentiment / headline analysis | **new** (`analysis/engagement.py`) |
| LLM reader-simulation judge | **new** (`analysis/llm_judge.py`) |
| Explainable AI layer (highlights, sources, reasoning) | **new** (`analysis/explainability.py`) |
| Weight calibration + benchmark | rewritten (`calibration/`), same 24-claim benchmark |

## Project structure

```
core/               # low-level model + API wrappers
  claim_detection.py     sentence segmentation, check-worthiness, AI probability
  evidence.py            Tavily web search + semantic relevance ranking
  verdict.py             Groq NLI verdict + deterministic risk formula
  rate_limit.py           shared 429 detection + backoff for Groq calls
  article_fetch.py        fetches + extracts article text from a URL
  metrics.py             readability + coherence
  embeddings.py          shared sentence-embedding model
analysis/           # higher-level scoring built on core/
  engagement.py          sentiment, lexical diversity, headline heuristics
  credibility.py          composite 0-100 credibility score
  llm_judge.py            LLM rates the article on 6 reader-perception dimensions
  explainability.py       turns raw output into a UI-ready XAI payload
calibration/        # weight calibration + evaluation for the risk formula
  benchmark_claims.py     24 labeled claims (true/false/unverifiable)
  weights.py               grid search (cached signals included)
  evaluate.py              precision/recall/F1/confusion matrix report
                           (claim tiers, reader study, AI-vs-human)
  ai_detector_eval.py       AI-text detector accuracy vs the paired
                           human/AI article set in study.db
  cache/raw_signals.json  a previous calibration run's cached signals
study/              # reader-perception study + analysis-history database
  database.py              SQLite: articles/sessions/ratings/behavior_events
                            (curated study set) + analyses/feedback
                            (every article analyzed through the API)
                            + survey delivery (assign_shown_label,
                            get_next_article_for_session, get_study_progress)
  api_routes.py             /study/* endpoints wrapping database.py for
                            the survey page (articles, session, next,
                            rating, behavior, progress)
  seed_articles.py           bulk-loads curated articles from .txt files
pipeline.py         # orchestrates everything end-to-end
api.py              # FastAPI backend: /analyze (text or URL), /history,
                     # /feedback, /analytics, /survey, /study/*
static/
  index.html               test frontend: Analyze / History / Analytics tabs
  study.html                reader-study survey page (served at /survey)
tests/
  test_pipeline.py        mocked end-to-end pipeline assembly tests
  test_rate_limit.py       Groq 429 detection + backoff tests
  test_article_fetch.py    URL fetch/extraction tests (mocked HTTP)
  test_study_survey.py     survey delivery: label assignment, exposure
                           balancing, progress tracking
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

You also need, **not included in this repo** (see `.gitignore`):

1. **The DeBERTa check-worthiness model folder**, named
   `deberta_claim_classifier_fp32_v2`, placed in the project root. Get this
   from whoever trained it - it's a large binary and was never committed.
2. **API keys.** Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`
   and `TAVILY_API_KEY`.

## Running

```bash
python api.py
# then open http://127.0.0.1:8000
```

Or use the pipeline directly:

```python
from pipeline import analyze_article
result = analyze_article("Your article text here...", title="Optional headline")
```

## URL input

`/analyze` accepts `article_text` OR `url` (the test UI has a "From URL"
tab). When given a URL, `core/article_fetch.py` downloads the page and
extracts the article text with `trafilatura` (falling back to a
BeautifulSoup paragraph scrape if trafilatura isn't installed or finds
nothing). Paywalled or JavaScript-rendered pages will fail extraction
with a clear 400 error rather than analyzing boilerplate/garbage text.

## Limiting claim verification time on long articles

Evidence retrieval + verdict scoring (stages 2-3) do one Tavily search
and one Groq call **per check-worthy claim**, sequentially, with a short
sleep between calls. A long article can easily have 50-100+ check-worthy
claims, which would take several minutes. `max_claims_to_verify`
(default 15, also exposed in the UI as "Max claims to verify") caps this:
only the top N claims **by check-worthiness (p_claim)** go through
evidence + verdict scoring; the rest are still listed in the results with
a `verification_skipped: true` flag and a visible reason, instead of
either being silently dropped or making every run take as long as the
worst-case article. `analysis/credibility.py`'s factual-reliability
component already only counts claims that have a `risk_score`, so
skipped claims correctly don't pull the credibility score in either
direction. Raise the cap (or set it very high) if you want every claim
checked and don't mind the wait - see `tests/test_pipeline.py`'s
`test_pipeline_caps_claims_verified_by_checkworthiness` for the exact
selection behavior.

## Reader feedback, history, and analytics

Every `/analyze` call is persisted to `study.db` (SQLite, created on
first run) with an `analysis_id`. The test UI then shows a 1-5 feedback
form (credibility/clarity/neutrality/complexity/coherence/quality) tied
to that id, posted to `/feedback`. `/history` lists recent analyses,
`/history/{id}` returns one in full (plus its feedback), and
`/analytics` aggregates score trends, credibility-tier distribution,
and mean reader feedback - all computed from real stored rows, with
`null`/empty fields rather than invented numbers when nothing has been
analyzed or rated yet.

This is separate from `study.database`'s `articles`/`ratings` tables,
which are reserved for the curated reader-study set with a known
ground-truth origin (human/ai) - see "Running the reader study" below.

## Groq rate-limit handling

`core/rate_limit.py` detects 429 responses from Groq (by status code,
exception class name, or message) in `core/verdict.py` and
`analysis/llm_judge.py`, and waits before retrying - honoring the
server's `Retry-After` header when present, otherwise exponential
backoff with jitter capped at 30s. Up to 5 rate-limited waits are
allowed per call, and none of them consume the existing `MAX_RETRIES`
budget for non-rate-limit errors, so prompts, model names, validation
logic, and fallback behavior are unchanged from the original code -
see `tests/test_rate_limit.py`.

## Running the tests (no model weights or API keys needed)

```bash
python tests/test_pipeline.py
python tests/test_rate_limit.py
python tests/test_article_fetch.py
python tests/test_study_survey.py
python tests/test_ai_detector_eval.py
python -m study.database
```

These mock out the DeBERTa/RoBERTa/Tavily/Groq calls (or use a temp
SQLite db / mocked HTTP) so you can check the pipeline, rate-limit
handling, URL fetching, and database layer all work correctly before
your API keys or model folder are in place.

## Recalibrating the risk weights

```bash
python -m calibration.weights                 # uses the cached signals, instant
python -m calibration.weights --refresh        # re-runs everything against your own keys
```

Current calibrated weights (`w_ai=0.05, w_claim=0.05, w_contradiction=0.90`)
score **83.33% tier accuracy** (20/24) on the 24-claim benchmark, vs 66.67%
for the untuned baseline.

## Evaluation / benchmarking report (Objective 5)

```bash
python -m calibration.evaluate              # prints the report
python -m calibration.evaluate --out report.json
```

Reports, from data that actually exists:
1. **Claim risk-tier classification** - full confusion matrix and
   per-class precision/recall/F1 against the 24-claim benchmark
   (currently 83.33% accuracy, macro F1 0.833).
2. **Reader study** - mean ratings per shown-label group, read live
   from `study.db`. Prints "no data yet" until the reader study
   (below) has actually been run - it does not fabricate numbers.
3. **AI-vs-human detection** - scores `Hello-SimpleAI/chatgpt-detector-roberta`
   against whatever paired human/AI articles are currently loaded into
   `study.db`'s curated set (see "Running the reader study" above - load
   `study_articles.zip`'s starter pairs with `study.seed_articles` first).
   Still reports "not available" rather than a number if fewer than 2
   articles of each origin are loaded. Can also be run on its own:
   ```bash
   python -m calibration.ai_detector_eval
   python -m calibration.ai_detector_eval --threshold 0.6 --out ai_detector_report.json
   ```
   Reports accuracy, confusion matrix, per-class precision/recall/F1,
   mean `p_ai_generated` by actual origin, and a per-article breakdown -
   explicitly labeled as a small-N pilot measurement, not a robust
   figure, until more pairs are loaded.

## Running the reader study

`study/database.py` is the schema for the curated study set (distinct
from the `/analyze` history above). There's now a full path to actually
run this with real participants:

**1. Prepare your articles.** Create a `study_articles/` folder (anywhere,
not inside the repo) with `human/` and `ai/` subfolders, one `.txt` file
per article - first line is the title, then a blank line, then the body:

```
study_articles/
  human/
    election_results.txt
  ai/
    election_results.txt   # same topic -> paired, strongest study design
```

**2. Load them into the database:**
```bash
python -m study.seed_articles study_articles/ --dry-run   # preview first
python -m study.seed_articles study_articles/
```
This calls `add_article(..., actual_origin="human"|"ai")` for each file.
`actual_origin` is the real, known origin - it is never sent to the
reader-facing survey.

**3. Send participants to the survey page:**
```
http://127.0.0.1:8000/survey
```
Each visitor gets an anonymous session id (stored in `localStorage`),
is served one not-yet-rated article at a time - picked to balance
exposure across the set via `get_next_article_for_session()` - under a
`shown_label` ("human" / "ai" / "none") assigned **deterministically**
per (session, article) by `assign_shown_label()`, independent of the
article's real origin. That's what lets you test whether the *label
alone* changes perception, separate from actual writing quality. The
page collects the six 1-5 ratings, plus `time_on_page` and
`scroll_depth` behavior events, via `POST /study/rating` and
`POST /study/behavior`.

**4. Read the results** with `get_all_ratings()` / `get_article_summary()`
(what `calibration/evaluate.py`'s `evaluate_reader_study()` reads from),
or query `study.db` directly - the `ratings` table joins each rating to
the article's real `actual_origin` *and* the `shown_label` the reader
saw, so you can compare both.

API reference for this flow: `POST /study/articles` (same as the seed
script, for adding one at a time), `GET /study/articles` (admin listing,
reveals `actual_origin`), `POST /study/session`, `GET /study/next`,
`POST /study/rating`, `POST /study/behavior`, `GET /study/progress`.

## Known limitations to state in the report

- **AI detector caveat.** `Hello-SimpleAI/chatgpt-detector-roberta` was
  trained on ChatGPT-vs-human short-form Q&A (HC3), not news. In our own
  check against the calibration benchmark, plain factual sentences scored
  ~0.97 mean `p_ai_generated` regardless of who wrote them - the detector
  appears to respond to formal writing register, not actual authorship.
  None of the 24 benchmark claims were actually AI-generated, so this
  isn't a measured AI-vs-human accuracy figure - it's a flag that you need
  a real paired human/AI dataset to measure that separately (Objective 5).
- **Engagement score is a text-only proxy.** Validate it against measured
  reader behaviour (dwell time, scroll depth, clicks) from the user study,
  don't report it as ground truth on its own.
- **Credibility weights are hand-set**, not fitted to real reader ratings
  yet. Objective 5 should regress human credibility ratings on the six
  component values and report the fitted weights.
