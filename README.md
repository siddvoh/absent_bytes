# Absent Bytes

Code and data for two papers on what medical vision-language models do when the image never arrives.

**Hearsay: Vision-Language Medical Diagnoses Without an Image.** Siddharth Vohra. TrustVLM Workshop at ICMR 2026.

**Absent-Byte Diagnoses: Auditing Structured Medical VLM Interfaces.** Siddharth Vohra, Manikandan Ravikiran. 2nd Agentic AI for Medicine Workshop at MICCAI 2026.

Both draw on one experiment. The first paper reports three models; the second extends the same grid to six and adds the interface audit.

## What the study does

A model is asked for a structured diagnosis with no image attached. The prompt varies only in how the patient is described: a neutral baseline, or one of twelve cells crossing age 32 or 65, three race descriptors and two sex descriptors, across chest X-ray, brain MRI and dermatology.

The question is whether the model declines, and when it does not, whether the diagnosis it invents moves with the wording. The second paper then asks whether a caller-side check can bind a response to the evidence actually sent and refuse the ones that cannot be bound.

## Layout

```
code/
  *.py             the library: sampling, schema, judging, metrics, normalisation,
                   the evidence contract and the contract oracle
  clients/         the six provider clients
  experiments/     everything executed for the first paper: collect/ analyze/ figures/
  scripts/         everything executed for the second paper: the interface audit
configs/           models, prompts, judge prompt, seeds, three taxonomies
data/
  *.json           the interface audit outputs
  records_E1/      234 files, 6 models x 3 domains x 13 conditions, 23,407 records
  categorized/     131 files, E1/E2/E2b/E4 for the first three models
  aggregated/      analysis outputs the figures and tables read
  derived/         the figures and tables the papers print
  evidence/        the field extracts and retained bodies the audits were computed from
  configs/         audit configuration
  manifests/       task manifests
third_party/       pinned upstream sources the interface audit reads
```

One library, at `code/`. Two folders of runnable code beside it, one per paper: `experiments/` collects and analyses the diagnosis grid, `scripts/` runs the interface audit that produces the second paper's tables. Everything under `data/` is either an input to those or an output of them.

`code/` and `code/scripts/` are the two names that are not free. `data/openclaw_holdout_freeze.json` pins six files by `code/scripts/...` path and sha256, backing the paper's claim that the OpenClaw holdout was chosen after the guard was frozen, and two of those six carry the string `code/` inside themselves. Renaming either one means editing a file whose hash is the thing being checked, so the library took the name that was already spoken for rather than adding a second one next to it.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Analysis, figures and tables, from the released records:

```bash
python code/experiments/analyze/run_analysis.py
python code/experiments/figures/generate_all_figures.py
python code/experiments/analyze/build_diagnosis_counts.py
python code/experiments/analyze/build_jsd_decomposition.py
python code/experiments/analyze/build_hedged_table.py
python code/experiments/analyze/build_e4_table.py
```

Interface audits:

```bash
python code/scripts/audit_enforcement_baselines.py --out data/enforcement_baseline_audit.json
python code/scripts/audit_independent_client_sources.py
python code/scripts/audit_direct_demographic_contrasts.py
python code/scripts/audit_boundary_guard_comparison.py --out data/boundary_guard_comparison_audit.json
python code/scripts/build_interface_figure.py
python code/scripts/build_demographic_pairs_figure.py
python code/scripts/build_jsd_figure.py
```

`audit_openclaw_holdout.py` also needs Node on the path. Everything writes into `data/derived/`.

To collect new data instead, run `run_e1.py`, then `run_mirage_judge.py`, then `run_analysis.py`. Anthropic, OpenAI and Gemini go direct. Qwen and Llama go through OpenRouter via `run_open_models_full.py`, which is budget-gated against the audit ledger. MedGemma runs locally on Colab from the notebook `build_medgemma_colab_notebook.py` emits.

## Versions

`requirements.txt` pins `google-genai`, `openai`, `httpx`, `jsonschema` and `pydantic` to the versions the released audits recorded. The enforcement and compositional audits hash request bodies built by those SDKs and refuse to run against different versions. That refusal is the point of the audit, not a bug.

## Data

`data/records_E1/SOURCE_MANIFEST.csv` gives filename, sha256 and record count for all 234 files. Every sha256 the second paper cites resolves to a file there.

227 files hold exactly 100 records. Seven medgemma files hold 101, each with one duplicated seed from a retry; both copies normalise to the same diagnosis, so counts are unaffected. Totals are 23,407 lines, 23,400 after deduplication.

`categorized/` is the raw record plus five judge fields. Two gemini records differ between the two sets: the categorized copy holds a connection error and `records_E1` holds the successful retry. Both are clean refusals.

There are two JSD matrices and they are not interchangeable. `aggregated/jsd_matrix.parquet` is 108 rows over the three models Hearsay reports. `aggregated/jsd_matrix_six_model.parquet` is 216 rows over all six and belongs to the second paper.

No patient data is present. Demographic descriptors are synthetic. The records are author-side application logs: they preserve the prompt, configuration, visible response and parsed output, not HTTP headers or provider request IDs.

## Two notes on the released files

`jsd_decomposition.csv` shipped with Jensen-Shannon distance in its `overall` column, from before `metrics.jsd` was changed to return divergence. `build_jsd_decomposition.py` regenerates it as divergence, consistent with every other table. Each changed cell is the square of the old value.

The Hearsay figure PDFs regenerate with the same numbers as the submitted ones but a lighter label weight, because the submitted pair was built with a bolder font configuration. Values, ratios and thresholds are identical.

## Licence

Code and data are MIT, `LICENSE`. Vendored sources under `third_party/` are Apache 2.0 and keep their own `LICENSE` and `PROVENANCE.txt` recording the upstream commit and hash.
