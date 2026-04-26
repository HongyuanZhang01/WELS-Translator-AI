"""
run_eval.py - Eval-only runner for already-translated jobs

This is the "second half" of the pipeline. It assumes chunks + translations
+ glossary already exist in a job folder (produced earlier by run_pipeline.py
with --skip-eval, or by run_batch.py), and runs the SAME evaluate -> improve
loop that run_pipeline.py's STEP 5+6 runs, with IDENTICAL behavior.

USAGE:
    python -m pipeline.run_eval \
        pipeline/data/batch_april2026/wauwatosa_spanish \
        --target Spanish \
        --source-language English \
        --context "Wauwatosa Theology Paper - English source" \
        --max-loops 3

WHAT IT READS (from the job folder):
    - exactly ONE chunks_*.json     (matched to the translations)
    - exactly ONE translations_*_{YYYYMMDD_HHMMSS}.json   (the work product)
    - exactly ONE glossary_{target_lower}.json

If any of those files are missing, not unique, or inconsistent with each
other (e.g. chunk count differs from translation count), the script ABORTS
IMMEDIATELY without touching anything. This is the safety net against
picking up half-finished debug artifacts.

WHAT IT WRITES (new files, originals are NEVER modified or deleted):
    - evaluated_translations_{target_lower}_{NEW_TIMESTAMP}.json
        (same schema as translations_*.json but with the BEST version of
         each chunk's text, after the improve loop runs)
    - evaluations_{target_lower}_{NEW_TIMESTAMP}.json
        (the raw rubric evaluations for each chunk)
    - improvement_log_{target_lower}_{NEW_TIMESTAMP}.json
        (record of every improvement attempt and whether it was accepted)
    - eval.log
        (human-readable progress log - tail this with `Get-Content -Wait`
         in PowerShell to watch progress live)

SAFETY GUARANTEES:
    1. Original translations_*.json is NEVER overwritten, renamed, or deleted.
    2. If the script aborts mid-run, whatever work is already done is still
       flushed to disk incrementally (eval results are appended one at a time).
    3. Exactly one input file per category or the script refuses to start.
    4. Glossary / chunk / translation schemas are validated before work begins.
"""

import os
import sys
import time
import json
import glob
import argparse
from datetime import datetime, timedelta

# UNICODE FIX (April 2026): On Windows, Python's stdout defaults to cp1252,
# which cannot encode characters outside Western European (e.g. Hungarian 'ő'
# U+0151, 'ű' U+0171). When the eval loop printed a parse-error message that
# happened to contain one of those characters, the whole subprocess crashed
# with UnicodeEncodeError and the eval results for that run were lost.
# Reconfiguring stdout to UTF-8 with errors='replace' makes printing any
# Unicode character safe: worst case, an un-displayable character shows as
# '?' in the terminal, but the process never crashes and file output (which
# already uses explicit encoding="utf-8") is unaffected.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    # Python < 3.7 doesn't have reconfigure; we require 3.7+ anyway.
    pass

sys.path.insert(0, os.path.dirname(__file__))

from eval.evaluator import full_evaluation
from improver import verified_improve, needs_improvement


class EvalLogger:
    """Writes every print() to both stdout and a log file, so you can tail
    the log file from PowerShell while still seeing progress on the terminal
    where the script was launched."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def find_exactly_one(folder, pattern, description):
    """Locate exactly one file matching `pattern` in `folder`. Abort if zero
    or more than one is found. This is the main safety gate that prevents
    us from accidentally reading an unfinished debug artifact."""
    matches = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(matches) == 0:
        raise SystemExit(
            f"ERROR: no {description} found in {folder} "
            f"(pattern: {pattern}). Cannot proceed."
        )
    if len(matches) > 1:
        raise SystemExit(
            f"ERROR: {len(matches)} {description} files found in {folder}, "
            f"expected exactly 1. Clean up the folder first so there is only "
            f"the real output file.\n"
            f"Found:\n  " + "\n  ".join(os.path.basename(m) for m in matches)
        )
    return matches[0]


def load_and_validate_json(path, expect_type, description):
    """Load a JSON file and check its top-level type. Abort on parse error
    or type mismatch — we do NOT guess our way through malformed input."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"ERROR: {description} at {path} is not valid JSON.\n"
            f"  Parse error: {e}"
        )
    if not isinstance(data, expect_type):
        raise SystemExit(
            f"ERROR: {description} at {path} has wrong top-level type: "
            f"expected {expect_type.__name__}, got {type(data).__name__}"
        )
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Run the evaluate->improve loop on an already-translated job folder",
    )
    parser.add_argument(
        "job_folder", type=str,
        help="Path to a job folder containing chunks_*.json, translations_*.json, "
             "and glossary_*.json (e.g. pipeline/data/batch_april2026/wauwatosa_spanish)",
    )
    parser.add_argument(
        "--target", type=str, required=True,
        help="Target language (must match what's in the translations file, "
             "e.g. Spanish or Hungarian)",
    )
    parser.add_argument(
        "--source-language", type=str, required=True,
        help="Source language of the original document (e.g. English, Latin, German)",
    )
    parser.add_argument(
        "--context", type=str, required=True,
        help="Document context string, same one that was passed to run_pipeline "
             "originally (e.g. 'Augsburg Confession by Philip Melanchthon, 1530')",
    )
    parser.add_argument(
        "--max-loops", type=int, default=3,
        help="Max improvement loops per chunk (default: 3). Set to 0 to disable.",
    )
    parser.add_argument(
        "--chunk-start", type=int, default=None,
        help="Start evaluating from this chunk number (1-indexed). For resuming.",
    )
    parser.add_argument(
        "--chunk-end", type=int, default=None,
        help="Stop evaluating after this chunk number (1-indexed).",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.job_folder)
    target_lower = args.target.lower()

    if not os.path.isdir(folder):
        raise SystemExit(f"ERROR: job folder does not exist: {folder}")

    # Start the log file BEFORE we do anything else, so the tail command picks
    # up our first message.
    log_path = os.path.join(folder, "eval.log")
    sys.stdout = EvalLogger(log_path)

    # Wall-clock start time for the whole run (used for elapsed reporting)
    run_start_time = time.monotonic()
    run_start_dt = datetime.now()

    timestamp = run_start_dt.strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("run_eval.py - Eval-Improve Loop on Existing Translations")
    print(f"Started: {run_start_dt.isoformat()}")
    print(f"Job folder: {folder}")
    print(f"Target: {args.target}")
    print(f"Source language: {args.source_language}")
    print(f"Max loops per chunk: {args.max_loops}")
    print("=" * 60)

    # =================================================================
    # STEP A: STRICT INPUT FILE SELECTION
    # Find exactly one of each required file. Abort if ambiguous.
    # =================================================================
    print("\n[A] Locating input files...")

    chunks_path = find_exactly_one(folder, "chunks_*.json", "chunks file")
    print(f"    chunks:       {os.path.basename(chunks_path)}")

    # Translations: only consider the pre-eval ones, NOT any evaluated_* we
    # might have written ourselves on a previous run.
    translations_path = find_exactly_one(
        folder, "translations_*.json", "translations file",
    )
    # Extra paranoia: make sure we didn't accidentally pick up an
    # evaluated_translations_*.json file (they start with "translations_"
    # if glob is loose, but ours are prefixed differently - still check).
    if "evaluated" in os.path.basename(translations_path):
        raise SystemExit(
            f"ERROR: picked up an evaluated_* file as the translations input: "
            f"{translations_path}. Clean up and retry."
        )
    print(f"    translations: {os.path.basename(translations_path)}")

    glossary_path = os.path.join(folder, f"glossary_{target_lower}.json")
    if not os.path.exists(glossary_path):
        raise SystemExit(
            f"ERROR: expected glossary file not found: {glossary_path}",
        )
    print(f"    glossary:     {os.path.basename(glossary_path)}")

    # =================================================================
    # STEP B: LOAD AND VALIDATE
    # =================================================================
    print("\n[B] Loading and validating inputs...")

    chunks = load_and_validate_json(chunks_path, list, "chunks")
    results = load_and_validate_json(translations_path, list, "translations")
    raw_glossary = load_and_validate_json(glossary_path, dict, "glossary")

    # NOTE: We deliberately do NOT abort on chunk/translation count mismatch,
    # missing translation keys, or target_language mismatch. run_pipeline.py
    # does not do those checks either — it handles count mismatch with
    # `chunks[i] if i < len(chunks) else {}` and lets missing keys KeyError
    # at the point of use. We match run_pipeline behavior exactly so the
    # eval-improve loop here is a drop-in replacement for STEP 5+6.

    # Flatten the glossary into the simple {term: translation} dict that
    # full_evaluation expects. This mirrors run_pipeline.py lines ~175-183.
    eval_glossary = {}
    for key, val in raw_glossary.items():
        if isinstance(val, dict):
            eval_glossary[key] = val.get("translation", "")
        else:
            eval_glossary[key] = val

    print(f"    chunks:       {len(chunks)}")
    print(f"    translations: {len(results)}")
    print(f"    glossary:     {len(eval_glossary)} terms")

    # =================================================================
    # STEP C: THE EVALUATE -> IMPROVE LOOP
    # This block is a verbatim port of run_pipeline.py STEP 5+6 (lines
    # ~185-320 in run_pipeline.py). Any behavior difference between the
    # two is a bug.
    # =================================================================
    print("\n" + "=" * 60)
    print("STEP C: EVALUATE -> IMPROVE LOOP")
    print("=" * 60)

    eval_results = []
    improvement_log = []

    # Optional chunk range filtering (for resuming)
    start_idx = (args.chunk_start - 1) if args.chunk_start else 0
    end_idx = args.chunk_end if args.chunk_end else len(results)
    work_items = list(enumerate(results))[start_idx:end_idx]

    if args.chunk_start or args.chunk_end:
        print(f"  Chunk range: {args.chunk_start or 1}-{args.chunk_end or len(results)} "
              f"(of {len(results)} total)")

    for loop_idx, (i, result) in enumerate(work_items):
        chunk_id = result["chunk_id"]
        current_text = result["translated_text"]
        source_text = result["source_text"]

        # Retrieve the chunk context the translator originally had. The
        # chunks list is ordered the same way results is, so index i works.
        chunk_obj = chunks[i] if i < len(chunks) else {}
        ctx_before = chunk_obj.get("context_before", "")
        ctx_after = chunk_obj.get("context_after", "")

        print(f"\n{'='*50}")
        print(f"  CHUNK {chunk_id} ({loop_idx+1}/{len(work_items)})")
        print(f"{'='*50}")

        # Track the best version we've seen (regression guard).
        best_text = current_text
        best_score = 0
        best_eval = None

        loop = 0
        while True:
            loop_label = f"pass {loop+1}" if loop == 0 else f"improvement {loop}"
            print(f"\n  [{loop_label}] Evaluating...")

            eval_result = full_evaluation(
                source_text=source_text,
                translated_text=current_text,
                target_language=args.target,
                chunk_id=chunk_id,
                glossary=eval_glossary,
                source_label=f"run_eval_loop{loop}",
            )

            rubric = eval_result.get("rubric_evaluation", {})
            da = rubric.get("doctrinal_accuracy", {}).get("score", 0)
            tc = rubric.get("terminology_consistency", {}).get("score", 0)
            ws = rubric.get("weighted_score", 0)

            print(f"  [{loop_label}] Doctrinal: {da}/5 | Terminology: {tc}/5 | Weighted: {ws}/5")

            # SCORE REGRESSION GUARD: keep the best version we've seen
            if ws > best_score:
                best_score = ws
                best_text = current_text
                best_eval = eval_result
            elif loop > 0 and ws < best_score:
                print(f"  [{loop_label}] REGRESSION DETECTED — score dropped "
                      f"from {best_score} to {ws}. Reverting to best version.")
                current_text = best_text
                eval_result = best_eval
                rubric = eval_result.get("rubric_evaluation", {})
                da = rubric.get("doctrinal_accuracy", {}).get("score", 0)
                tc = rubric.get("terminology_consistency", {}).get("score", 0)
                ws = rubric.get("weighted_score", 0)

            if not needs_improvement(eval_result):
                print(f"  [{loop_label}] TARGET MET — doctrinal {da}/5, terminology {tc}/5")
                break

            if loop >= args.max_loops or args.max_loops == 0:
                print(f"  [{loop_label}] Max loops reached ({args.max_loops}). "
                      f"Best scores: doctrinal {da}/5, terminology {tc}/5")
                break

            loop += 1
            print(f"\n  [improvement {loop}] Feeding feedback to improver (with full context)...")
            errors_count = len(rubric.get("critical_errors", []))
            suggestions_count = len(rubric.get("suggestions", []))
            print(f"  [improvement {loop}] Fixing {errors_count} errors + {suggestions_count} suggestions...")

            improve_result = verified_improve(
                source_text=source_text,
                translated_text=current_text,
                eval_result=eval_result,
                target_language=args.target,
                glossary=eval_glossary,
                source_language=args.source_language,
                document_context=args.context,
                context_before=ctx_before,
                context_after=ctx_after,
                max_attempts=2,
            )

            improvement_log.append({
                "chunk_id": chunk_id,
                "loop": loop,
                "before_scores": {
                    "doctrinal_accuracy": da,
                    "terminology_consistency": tc,
                    "weighted_score": ws,
                },
                "issues_addressed": errors_count + suggestions_count,
                "accepted": improve_result["accepted"],
                "attempts": improve_result["attempts"],
                "rejection_reasons": improve_result.get("rejection_reasons", []),
            })

            if not improve_result["accepted"]:
                print(f"  [improvement {loop}] Improvement rejected after "
                      f"{improve_result['attempts']} attempts. Keeping current translation.")
                break

            current_text = improve_result["improved_text"]
            result["translated_text"] = current_text

        # Make sure we save the BEST version, not just the last
        result["translated_text"] = best_text if best_eval else current_text
        eval_results.append(best_eval if best_eval else eval_result)

        # ================================================================
        # INCREMENTAL FLUSH: write current state after every chunk, so if
        # the process crashes partway through we still have results for
        # everything finished so far.
        # ================================================================
        eval_translations_path = os.path.join(
            folder, f"evaluated_translations_{target_lower}_{timestamp}.json",
        )
        with open(eval_translations_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        eval_path = os.path.join(
            folder, f"evaluations_{target_lower}_{timestamp}.json",
        )
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)

        if improvement_log:
            log_path_json = os.path.join(
                folder, f"improvement_log_{target_lower}_{timestamp}.json",
            )
            with open(log_path_json, "w", encoding="utf-8") as f:
                json.dump(improvement_log, f, indent=2, ensure_ascii=False)

    # =================================================================
    # STEP D: SUMMARY
    # =================================================================
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    scores = [
        r["rubric_evaluation"].get("weighted_score", 0)
        for r in eval_results
        if "error" not in r.get("rubric_evaluation", {})
    ]
    if scores:
        print(f"  Chunks evaluated:    {len(results)}")
        print(f"  Improvement passes:  {len(improvement_log)}")
        print(f"  Average score:       {sum(scores)/len(scores):.2f}/5.0")
        print(f"  Highest:             {max(scores)}/5.0")
        print(f"  Lowest:              {min(scores)}/5.0")
        print()
        for er in eval_results:
            r = er.get("rubric_evaluation", {})
            cid = er.get("chunk_id", "?")
            da = r.get("doctrinal_accuracy", {}).get("score", "?")
            tc = r.get("terminology_consistency", {}).get("score", "?")
            print(f"    Chunk {cid}: doctrinal={da}/5, terminology={tc}/5")

    # =================================================================
    # FLAGGED FOR HUMAN REVIEW
    # If both dual-pass rubric attempts failed (Pass A AND Pass B both
    # returned unparseable JSON), the evaluator marks the chunk with
    # needs_human_review=True. We surface these loudly at the end of the
    # run so a human knows exactly which chunks cannot be trusted and
    # need manual inspection.
    # =================================================================
    flagged = [er for er in eval_results if er.get("needs_human_review")]
    if flagged:
        print()
        print("!" * 60)
        print(f"!!! {len(flagged)} CHUNK(S) FLAGGED FOR HUMAN REVIEW !!!")
        print("!" * 60)
        print("  These chunks could not be automatically evaluated because")
        print("  BOTH dual-pass rubric attempts (Pass A and Pass B) failed")
        print("  to return parseable JSON. A human must review them manually.")
        print()
        for er in flagged:
            cid = er.get("chunk_id", "?")
            reason = er.get("review_reason", "Unknown — check the evaluations JSON.")
            failed = er.get("failed_passes", [])
            failed_str = ", ".join(failed) if failed else "unknown"
            print(f"  ⚠  Chunk {cid}  (failed passes: {failed_str})")
            print(f"     {reason}")
        print("!" * 60)

    print()
    print(f"  Final translations:  {os.path.basename(eval_translations_path)}")
    print(f"  Evaluations:         {os.path.basename(eval_path)}")
    if improvement_log:
        print(f"  Improvement log:     improvement_log_{target_lower}_{timestamp}.json")
    print(f"  Original translations UNTOUCHED: {os.path.basename(translations_path)}")

    # Wall-clock elapsed time reporting
    run_end_dt = datetime.now()
    elapsed_seconds = time.monotonic() - run_start_time
    elapsed_td = timedelta(seconds=int(elapsed_seconds))
    chunks_done = len(eval_results)
    per_chunk = (elapsed_seconds / chunks_done) if chunks_done else 0.0
    print(f"  Started:  {run_start_dt.isoformat()}")
    print(f"  Finished: {run_end_dt.isoformat()}")
    print(f"  Elapsed:  {elapsed_td} ({elapsed_seconds:.1f} seconds)")
    if chunks_done:
        print(f"  Per chunk avg: {per_chunk:.1f} seconds  ({chunks_done} chunks processed)")
    print("=" * 60)


if __name__ == "__main__":
    main()
