# Handoff to Alexander — WELS Translation Pipeline

**From:** Joseph
**Date:** 2026-04-26
**Repo:** https://github.com/HongyuanZhang01/WELS-Translator-AI

---

## What you're getting

The whole `pipeline/` folder is the current translation system, end to end.
It takes a source document (Latin / English / German), chunks it, builds a
glossary, translates with Claude Opus, evaluates with Claude Sonnet, runs
an improvement loop until scores stabilize, and packages four deliverable
files for human review. See `PIPELINE_FLOWCHART.md` in this repo for the
visual mermaid diagram.

For the front end, the most important things to know are:

1. **The pipeline is a CLI right now.** Every step is invoked via
   `python -m pipeline.run_pipeline ...`. Wrapping that in a web UI is
   the front-end task. The pipeline's existing CLI flags (target language,
   chunk range, max-loops, skip-eval, output dir) map cleanly onto form
   fields in a UI.

2. **Every run produces files with a shared timestamp.** Files named
   `chunks_20260411_140000.json`, `translations_hungarian_20260411_140000.json`,
   `evaluations_hungarian_20260411_140000.json`, etc. all belong to one job.
   The front end should let users browse runs grouped by that timestamp.

3. **The translator writes incrementally.** As each chunk finishes, the
   `translations_*.json` file is rewritten with one more entry. So a UI
   could safely read this file mid-run to display progress without waiting
   for the whole job.

4. **Bible lookups can fail.** When that happens, a sidecar file
   `bible_lookup_review_<target>_<timestamp>.json` is written. The front
   end should surface this prominently — it means a human translator
   needs to verify those verses before delivery.

5. **Outputs are designed for human review.** `package_translations.py`
   produces a `chunks_needing_human_review.txt` file that lists only
   the flagged chunks. That's the primary output reviewers (like Pastor
   Julio for Spanish) actually read.

---

## Key files to look at first

| File | Why it matters for the front end |
|------|----------------------------------|
| `pipeline/run_pipeline.py` | The full pipeline orchestrator. The CLI flags here are the UI's input fields. |
| `pipeline/run_eval.py` | The "second half" re-evaluates an existing translation without re-translating. Useful if a user wants to change the rubric and re-score. |
| `pipeline/config.py` | Model names, target languages, evaluation weights. A UI settings panel could expose `EVAL_WEIGHTS`. |
| `pipeline/package_translations.py` | Builds the four deliverable files. The front end will probably want to call this to render results. |
| `pipeline/eval/evaluator.py` | The dual-pass rubric. Has detailed docstrings about how scores are computed. |

---

## Concerns / open issues you should know about

These are non-critical errors Joseph hit during the April testing batch that Jonathan has right now
(otto / augsburg / catechism × Spanish / Hungarian). Some have been fixed
in code, others are still open. None are blockers for starting the front
end, but they may affect what you build.

### Pipeline-side concerns

1. **Chunker reliability is uneven.**
   - On the Otto Convention Essay, the chunker silently dropped roughly
     43% of the source text — only 56.6% coverage. We caught it after the
     fact during packaging. The chunker now has a 90% coverage tripwire
     that refuses to write the chunks file, but that fix only catches
     this particular failure mode. The chunker also produced
     heading-only stub chunks (e.g. just `"Article 1, God"` with no
     body), which wasted API calls and triggered hallucinations in the
     translator. I fixed this by merging stubs into the next chunk.
   - Augsburg article boundary detection on the Latin source mislabeled
     51 of 78 body chunks as "Article 28" because the chunker couldn't
     find Latin article markers (`Articulus I`, `II`, ...). The
     translation itself is fine because the translator reads
     `source_text` directly, but the rubric-side article framing was
     noisy.
   - **What this means for the front end:** Every run should display
     a coverage % (extracted text length vs. source) and let users
     reject runs below a threshold before the translator burns API
     credits.

2. **The translator can hallucinate when given too little input.**
   On a 14-character source ("Article 1, God"), the translator wrote
   1130 chars of theology pulled from training data — looked plausible,
   was not in the source. We added a guard against absurd output-length
   ratios and against heading-only inputs, but a UI should still
   surface "translation length vs. source length" as a warning signal.

3. **Bible-quote handling is done but conservative.**
   The pipeline tries to look up the published target-language Bible
   text for every Bible reference it detects. When the API refuses
   (content-filter false positives, transient errors), it falls back
   to inline AI translation and writes the failed lookups to a JSON
   sidecar. For front end, I think it should not auto-publish translations
   while that sidecar has unresolved entries.

4. **Consistency across chunks.**
   Benjamin pointed out (review of the Hungarian Augsburg) that chunks
   used "Cikk" and "Cikkely" inconsistently for "Article" earlier. We now pass
   the previous chunk's translation as a "consistency anchor" into
   the next chunk's prompt, which helps a lot but isn't perfect. The
   glossary catches theological terms, the anchor catches structural
   conventions, but truly long-document consistency may need a
   post-pass terminology audit. The front end might want a
   "terminology audit" panel that shows which terms varied across
   the document.

5. **Training-data contamination on famous texts.**
   The Augsburg Confession is widely available in many languages on
   the open web, so the model may be "remembering" rather than
   "translating." For evaluating the pipeline on its own merits,
   less-famous texts (the Wauwatosa Paper, the Otto Essay) are
   better operational tests. This is worth being aware of when interpreting
   scores from the front end's display.

### Process-side concerns

6. **The improvement loop can be slow on hard chunks.**
   The default max-loops is 3, with each loop doing 1 evaluation +
   up to 2 verified-improvement attempts. On a chunk that the
   evaluator keeps disagreeing with, that's 3 evaluations + 6 improvement
   attempts = 9 API calls just to land on "best of bad options." For
   a 70-chunk document like the Augsburg, the worst case is ~600
   API calls. The front end should expose a per-job cost estimate
   before kicking off and a per-chunk cost log after.

7. **The whole workflow assumes one document at a time.**
   We don't currently batch multiple documents in a single run. If
   the front end is going to support a "translate the entire Book of
   Concord" workflow, the orchestration layer for that is still TBD.

### Tech / hygiene concerns

8. **No automated tests yet.**
   Everything has been validated by running the pipeline on real
   documents and human-reviewing the output. There's no pytest
   suite. Worth flagging because changes you make to the pipeline
   to support the UI could silently break translation quality.

9. **No API-cost tracker.**
    We've been eyeballing costs from the Anthropic console. A simple
    in-pipeline token counter that writes to `cost_log.json` per run
    would make the UI's billing display much easier to build later.

10. **State files are JSON, not a database.**
    Easy to reason about, easy to back up, but not great for
    concurrent access. If the UI lets two people run the same job
    simultaneously, they will race on the same output directory.
    Perhaps we can either lock the output dir, or assign a per-user output namespace.

---

## Things that would be beneficial to include in the front-end:

- **Per-chunk red/yellow/green** based on doctrinal_accuracy + terminology_consistency.
- **Side-by-side source / translation viewer** with the chunk's
  evaluation feedback rendered next to it.
- **A "human review" annotation layer** so future
  reviewers can comment per chunk, and those comments live with
  the translation in the repo, along with being then manually ask
  the pipeline to translate this one chunk again immediately,
  with the new feedback from the user.
- **Run history** for each job, grouped by source doc and target language, with
  the timestamp filter.
- **A "what does the rubric mean" tooltip** anywhere a score is
  shown, because currently the four dimensions and their weights 
  are not obvious to non-translators.

---

## How to run the pipeline (sanity check)

```bash
pip install -r pipeline/requirements.txt
export ANTHROPIC_API_KEY="your-key-here"

# Small smoke test if you want (3 chunks, no eval loop):
python -m pipeline.run_pipeline pipeline/data/sample.txt \
    --target Spanish --max-chunks 3 --skip-eval
```

If that runs clean and produces a `translations_spanish_*.json`, the
pipeline is wired up on your machine. From there you can scale up to
the eval loop and the full document.

---

