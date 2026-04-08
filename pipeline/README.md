# WELS Translation Pipeline

AI-powered translation pipeline for WELS Lutheran confessional documents. Translates, evaluates, and iteratively improves translations using Claude.

## How it works

Source document → Chunk by article → Build glossary → Detect Bible quotes → Translate (Claude Opus) → Evaluate with dual-pass rubric (Claude Sonnet) → Improve if needed → Save final output

See `Current_Pipeline_flowchart.png` for the visual diagram.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python -m pipeline.run_pipeline pipeline/data/augsburg_confession_english.txt --target Hungarian
```

Useful flags:
- `--chunk-start 4 --chunk-end 13` — process only a range of chunks
- `--max-loops 3` — max improvement attempts per chunk (default: 3)
- `--skip-eval` — translate without evaluation (faster, no scores)

## Generate Excel report

```bash
python -m pipeline.eval.export_to_excel --file pipeline/data/evaluations_hungarian_XXXXX.json --output report.xlsx
```

## Current results (Hungarian, Augsburg Confession)

- 70 chunks, 28 articles, full document translated
- Average weighted score: **4.72 / 5.0**
- Doctrinal accuracy and terminology: nearly all 5.0
- See `data/evaluation_report_hungarian_FULL.xlsx` for the full breakdown

## Key files

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Main entry point — runs the full pipeline |
| `config.py` | Model settings, evaluation weights |
| `chunker.py` | Splits documents by article boundaries |
| `glossary.py` | Extracts and translates theological terms |
| `translator.py` | Translates chunks with full context |
| `improver.py` | Verified improvement loop with glossary regression detection |
| `eval/evaluator.py` | Dual-pass rubric evaluation + back-translation |
| `eval/export_to_excel.py` | Generates Excel reports from evaluation JSON |
