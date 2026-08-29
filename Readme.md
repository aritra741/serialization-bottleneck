# Experiment 1 — Property Recovery from Text-Serialized Structured Objects

Does serialization format cost a model accuracy? This repository holds
Experiment 1 across three domains so far: Geometry (polygons), Graphs, and
Tabular (CSV tables). Each domain is a 300-object benchmark, six model runs
over it, and the evaluation of those runs.

Reference spec: `serialization_experiment_1.pdf`.

## Pipeline

### Domain 1 — Geometry

| Phase | Folder | What it does | Docs |
|-------|--------|--------------|------|
| 1 | `phase1_dataset/` | Generates the 300-polygon dataset with 9 ground-truth properties | [README](phase1_dataset/README.md) |
| 2 | `phase2_model_results/` | Queries 6 models — 9 properties per polygon — and scores the answers | [README](phase2_model_results/README.md) |
| 3 | `phase3_evaluation/` | Aggregates the Phase-2 records into metrics and figures | [README](phase3_evaluation/README.md) |

### Domain 2 — Graphs

| Phase | Folder | What it does | Docs |
|-------|--------|--------------|------|
| 1 | `phase1_dataset_graph/` | Generates the 300-graph dataset with 8 ground-truth properties | [README](phase1_dataset_graph/README.md) |
| 2 | `phase2_model_results_graph/` | Queries 6 models — 8 properties per graph — and scores the answers | [README](phase2_model_results_graph/README.md) |
| 3 | `phase3_evaluation_graph/` | Aggregates the Phase-2 records into metrics and figures | [README](phase3_evaluation_graph/README.md) |

### Domain 3 — Tabular

CSV tables instead of polygons or graphs: 8 ground-truth properties split
into *local* ones readable off a single cell/column (row count, column
dtype, null count) and *global* ones requiring whole-column aggregation
(Pearson correlation, monotonicity, outliers, skewness, functional
dependency) — Phase 3 measures the accuracy gap between the two as the
project's core diagnostic. Unlike the other two domains, Phase 1 and
Phase 3 here are plain scripts rather than notebooks, and Phase 2 uses one
shared `harness.py` instead of six separate `run_*_full.py` files.

| Phase | Folder | What it does |
|-------|--------|--------------|
| 1 | `phase1_tabular_dataset/` | Generates the 300-table dataset (`dataset.py`) with 8 ground-truth properties, independently re-verified from the raw CSV text |
| 2 | `phase2_tabular_model_results/` | Queries the same 6 models via a shared `harness.py` — up to 10 properties per table — and scores the answers |
| 3 | `phase3_tabular_evaluation/` | Aggregates the Phase-2 records into metrics (`evaluate.py`) and figures (`figures.py`), including the local/global accuracy gap and its bootstrap significance test |

**Status: fully run.** All 6 models queried on the real dataset (5 at full
2,953-query coverage, DeepSeek-V4-Pro-thinking on a 588-query 20%
subsample), 0 API failures across every model. Every model shows a large,
statistically significant local-global accuracy gap (24-54 percentage
points, p<0.0001) — `correlation` and `skewness` are the universally
hardest properties (0-5% accuracy across all six models), `column_dtype`
the easiest (96-100%). Overall accuracy ranges from 50.4% (Qwen3-32B) to
73.6% (DeepSeek-V4-Pro thinking mode). Full breakdown in
`phase3_tabular_evaluation/evaluation_summary.txt` and
`evaluation_report.json`. The 50 real-data table slots (of 300) are still
unsourced pending a human license/novelty check — see
`phase1_tabular_dataset/real_data_sources/README.md`; the current build is
300/300 synthetic. The PDF §7 qualitative failure-analysis artifact is not
yet built.

Each phase consumes the previous one's output files within its domain, so
they run in order. Phase 1 is fully offline in all three domains; only
Phase 2 calls APIs.

---

## Setup

Covers **all three phases** — one virtual environment, one install.

Python 3.10 or newer is required (the Phase-2 runners use `list[dict[str, Any]]`
/ `X | Y` type syntax).

```bash
cd /path/to/EXPERIMENT1_PHASE1-2

python3 -m venv .venv                 # create the virtual environment
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` covers every phase:

| Package | Used by |
|---------|---------|
| `jupyter>=1.0` | Phase 1, Phase 3 — running the notebooks |
| `shapely>=2.0` | Phase 1 (Geometry) — polygon construction and ground truth |
| `networkx>=3.2` | Phase 1 (Graphs) — graph construction and ground truth |
| `pandas>=2.0` | Phase 1 (Tabular) — table construction, dtype inference, ground truth |
| `matplotlib>=3.7` | Phase 1 — the spot-check figures; Phase 3 — the figures |
| `openai>=1.30` | Phase 2 — the API client (all providers) |
| `python-dotenv>=1.0` | Phase 2 — loads `.env` |
| `tenacity>=8.2` | Phase 2 — retry with exponential backoff |
| `tqdm>=4.66` | Phase 2 — progress bars |
| `numpy>=1.24` | Phase 3 — metric aggregation |
| `scipy>=1.10` | Phase 3 — statistics |

### API keys

Only Phase 2 calls APIs, but the credentials are set up here so both phases are
ready in one pass:

```bash
cp .env.example .env       # then edit .env and fill in the keys you need
```

| Key | Provider | Used by |
|-----|----------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek direct | `01_v4flash`, `06_v4pro_thinking` (Tabular domain) / `06_v4pro_nonthinking` (Geometry/Graphs) |
| `OPENROUTER_API_KEY` | OpenRouter | `02_qwen3`, `03_llama_scout` |
| `GEMINI_API_KEY` | Google | `04_gemini` |
| `OPENAI_API_KEY` | OpenAI | `05_gpt` |

You only need the key for a provider you actually intend to run. A missing key
fails at startup with `<KEY_NAME> not set. Add it to .env.`

`.env` is gitignored; `.env.example` is not. Never commit real keys.

The Phase-2 runners call `load_dotenv()` with no argument, which searches upward
from the script's own directory — so `.env` at this root is found from inside any
runner folder.

### Verify the setup

Two commands, both free and neither needing an API key:

```bash
# Phase 1 — open the notebook and run the cells in order
jupyter notebook phase1_dataset/Geometry_Experiment1_Phase1.ipynb
# Domain 2 (Graphs) Phase 1 works the same way:
jupyter notebook phase1_dataset_graph/Graph_Experiment1_Phase1.ipynb
# Domain 3 (Tabular) Phase 1 is a plain script instead of a notebook:
python phase1_tabular_dataset/dataset.py

# Phase 2 — build prompts without calling anything
cd phase2_model_results/01_v4flash
python run_v4flash_full.py \
  --dataset ../../phase1_dataset/geometry_exp1_dataset.json \
  --jsonl-output v4flash_results.jsonl \
  --json-output v4flash_results.json \
  --dry-run

# Domain 3 (Tabular) Phase 2 uses one shared harness.py instead of a
# per-model run_*_full.py:
cd ../../phase2_tabular_model_results/01_v4flash
python run.py \
  --dataset ../../phase1_tabular_dataset/tabular_exp1_dataset.json \
  --jsonl-output v4flash_results.jsonl \
  --json-output v4flash_results.json \
  --dry-run
```

The Phase-2 dry run prints three example prompts and the planned query count.
With the committed results in place it reports `skipping 2700 already done` —
which also confirms the resume logic is reading the existing result files.

The two output flags are required for that: without them the script looks in a
`results/` subfolder that does not exist, finds no history, and reports 2,700
queries planned. See `phase2_model_results/README.md` §5 and §9 for why.

### Phase 3 dependencies

Phase 3 also ships its own `phase3_evaluation/requirements.txt` (numpy, scipy,
matplotlib) so that folder stays self-contained. Those packages are included in
the root `requirements.txt` above, so installing once at the root is enough — the
per-folder file is only needed if Phase 3 is run in isolation.

---

## Where to go next

- Regenerating or porting the Geometry dataset → [`phase1_dataset/README.md`](phase1_dataset/README.md)
- Running the Geometry models, or porting the query harness → [`phase2_model_results/README.md`](phase2_model_results/README.md)
- Regenerating or porting the Graph dataset → [`phase1_dataset_graph/README.md`](phase1_dataset_graph/README.md)
- Running the Graph models, or porting the query harness → [`phase2_model_results_graph/README.md`](phase2_model_results_graph/README.md)
- Graph evaluation notebook (not yet run against real API data) → [`phase3_evaluation_graph/README.md`](phase3_evaluation_graph/README.md)
- Regenerating or porting the Tabular dataset → [`phase1_tabular_dataset/dataset.py`](phase1_tabular_dataset/dataset.py) (module docstring)
- Running the Tabular models, or porting the query harness → [`phase2_tabular_model_results/harness.py`](phase2_tabular_model_results/harness.py) (module docstring)
- Tabular evaluation results (all 6 models, real data) → [`phase3_tabular_evaluation/evaluation_summary.txt`](phase3_tabular_evaluation/evaluation_summary.txt)

Each dataset README (or, for the Tabular domain, module docstring) ends
with a porting section describing what is domain-agnostic and what a new
domain has to replace.
