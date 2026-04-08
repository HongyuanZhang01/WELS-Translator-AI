"""
run_pipeline.py - Main Pipeline Runner

Runs the full translation pipeline end-to-end:
  Chunk -> Glossary -> Translate -> Evaluate -> Improve (auto-loop)

USAGE:
  python -m pipeline.run_pipeline input.txt --target Spanish
  python -m pipeline.run_pipeline input.pdf --target Hungarian --context "Augsburg Confession, Article IV"
  python -m pipeline.run_pipeline input.txt --target Hmong --max-chunks 3 --skip-eval
  python -m pipeline.run_pipeline input.txt --target Hungarian --max-loops 3
  python -m pipeline.run_pipeline input.txt --target Hungarian --chunk-start 4 --chunk-end 13

The auto-loop: after evaluating, if doctrinal accuracy or terminology
consistency is below 5/5, the pipeline feeds the evaluation feedback
back to the translator and tries again. It keeps looping until both
hit 5/5 or the max number of loops is reached (default: 3).
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import SOURCE_LANGUAGE, DEFAULT_TARGET
from chunker import ingest_document, detect_articles, chunk_text, save_chunks
from translator import translate_chunks
from eval.evaluator import full_evaluation
from improver import verified_improve, needs_improvement


def main():
    parser = argparse.ArgumentParser(
        description="Run the full WELS translation pipeline"
    )
    parser.add_argument(
        "input", type=str,
        help="Path to the source document (.txt, .pdf, or .json)"
    )
    parser.add_argument(
        "--target", type=str, default=DEFAULT_TARGET,
        help=f"Target language (default: {DEFAULT_TARGET})"
    )
    parser.add_argument(
        "--context", type=str, default=None,
        help="Document context (e.g., 'Augsburg Confession, Article IV on Justification')"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1500,
        help="Chunk size in characters (default: 1500)"
    )
    parser.add_argument(
        "--overlap", type=int, default=200,
        help="Chunk overlap in characters (default: 200)"
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None,
        help="Limit number of chunks (for testing)"
    )
    parser.add_argument(
        "--chunk-start", type=int, default=None,
        help="Start processing from this chunk number (1-indexed, inclusive). "
             "E.g., --chunk-start 4 skips chunks 001-003."
    )
    parser.add_argument(
        "--chunk-end", type=int, default=None,
        help="Stop processing after this chunk number (1-indexed, inclusive). "
             "E.g., --chunk-end 13 stops after chunk 013."
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Source language override"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip the evaluation step (faster, but no quality scores)"
    )
    parser.add_argument(
        "--max-loops", type=int, default=3,
        help="Max improvement loops per chunk (default: 3). Set to 0 to disable."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for all output files"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_lower = args.target.lower()
    data_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    source_lang = args.language or SOURCE_LANGUAGE

    # =================================================================
    # STEP 1: INGEST AND CHUNK
    # =================================================================
    print("=" * 60)
    print("STEP 1: INGEST AND CHUNK")
    print("=" * 60)

    doc = ingest_document(args.input, source_lang)
    print(f"  Document: {doc['filename']}")
    print(f"  Language: {doc['language']}")
    print(f"  Length: {len(doc['text'])} characters")

    articles = detect_articles(doc["text"])
    if articles:
        print(f"  Structure: {len(articles)} articles/sections detected")

    chunks = chunk_text(
        doc["text"],
        chunk_size=args.chunk_size,
        overlap_size=args.overlap,
        articles=articles if articles else None,
    )

    if args.max_chunks:
        chunks = chunks[:args.max_chunks]

    # Chunk range filtering: --chunk-start and --chunk-end
    # These use 1-indexed chunk numbers matching the chunk_id field.
    # The full chunk list is generated first (so context_before/after
    # are correct), then we slice to just the range we want.
    if args.chunk_start or args.chunk_end:
        start_idx = (args.chunk_start - 1) if args.chunk_start else 0
        end_idx = args.chunk_end if args.chunk_end else len(chunks)
        total_before = len(chunks)
        chunks = chunks[start_idx:end_idx]
        print(f"  Chunk range: {args.chunk_start or 1}-{args.chunk_end or total_before} "
              f"(filtered from {total_before} total)")

    print(f"  Chunks: {len(chunks)} to process")

    chunks_path = os.path.join(data_dir, f"chunks_{timestamp}.json")
    save_chunks(chunks, chunks_path)

    # =================================================================
    # STEPS 3 + 4: GLOSSARY + TRANSLATE
    # =================================================================
    print("\n" + "=" * 60)
    print(f"STEPS 3+4: GLOSSARY + TRANSLATE -> {args.target}")
    print("=" * 60)

    glossary_path = os.path.join(data_dir, f"glossary_{target_lower}.json")

    # Build document context
    doc_context = args.context
    if not doc_context:
        doc_context = f"Source document: {doc['filename']}"

    results = translate_chunks(
        chunks=chunks,
        target_language=args.target,
        glossary_path=glossary_path,
        source_language=source_lang,
        document_context=doc_context,
    )

    translations_path = os.path.join(data_dir, f"translations_{target_lower}_{timestamp}.json")
    with open(translations_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Translations saved to {translations_path}")

    # =================================================================
    # STEP 5 + 6: EVALUATE -> IMPROVE LOOP
    # =================================================================
    if not args.skip_eval:
        print("\n" + "=" * 60)
        print("STEP 5+6: EVALUATE -> IMPROVE LOOP")
        print("=" * 60)

        # Load the glossary as a simple dict for the evaluator
        eval_glossary = None
        if os.path.exists(glossary_path):
            with open(glossary_path, "r", encoding="utf-8") as f:
                raw_glossary = json.load(f)
            eval_glossary = {}
            for key, val in raw_glossary.items():
                if isinstance(val, dict):
                    eval_glossary[key] = val.get("translation", "")
                else:
                    eval_glossary[key] = val

        eval_results = []
        improvement_log = []

        for i, result in enumerate(results):
            chunk_id = result["chunk_id"]
            current_text = result["translated_text"]
            source_text = result["source_text"]

            # Retrieve the chunk context the translator originally had
            chunk_obj = chunks[i] if i < len(chunks) else {}
            ctx_before = chunk_obj.get("context_before", "")
            ctx_after = chunk_obj.get("context_after", "")

            print(f"\n{'='*50}")
            print(f"  CHUNK {chunk_id} ({i+1}/{len(results)})")
            print(f"{'='*50}")

            # Track the best version we've seen (for regression guard)
            best_text = current_text
            best_score = 0
            best_eval = None

            # Evaluate-improve loop for this chunk
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
                    source_label=f"pipeline_loop{loop}",
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
                    # Don't break — still check if target is met or loops remain

                # Check if we've hit the target or max loops
                if not needs_improvement(eval_result):
                    print(f"  [{loop_label}] TARGET MET — doctrinal {da}/5, terminology {tc}/5")
                    break

                if loop >= args.max_loops or args.max_loops == 0:
                    print(f"  [{loop_label}] Max loops reached ({args.max_loops}). "
                          f"Best scores: doctrinal {da}/5, terminology {tc}/5")
                    break

                # Improve with full context + verification
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
                    source_language=source_lang,
                    document_context=doc_context,
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
                # Update the translation in results
                result["translated_text"] = current_text

            # Make sure we save the BEST version, not just the last
            result["translated_text"] = best_text if best_eval else current_text
            eval_results.append(best_eval if best_eval else eval_result)

        # Save final translations (with improvements applied)
        final_translations_path = os.path.join(
            data_dir, f"translations_{target_lower}_{timestamp}_final.json"
        )
        with open(final_translations_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Final translations saved to {final_translations_path}")

        # Save evaluations
        eval_path = os.path.join(data_dir, f"evaluations_{target_lower}_{timestamp}.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print(f"  Evaluations saved to {eval_path}")

        # Save improvement log
        if improvement_log:
            log_path = os.path.join(data_dir, f"improvement_log_{target_lower}_{timestamp}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(improvement_log, f, indent=2, ensure_ascii=False)
            print(f"  Improvement log saved to {log_path}")

        # Print summary
        scores = [
            r["rubric_evaluation"].get("weighted_score", 0)
            for r in eval_results
            if "error" not in r.get("rubric_evaluation", {})
        ]
        if scores:
            print(f"\n  RESULTS SUMMARY:")
            print(f"    Chunks translated:     {len(results)}")
            print(f"    Improvement passes:    {len(improvement_log)}")
            print(f"    Average final score:   {sum(scores)/len(scores):.2f}/5.0")
            print(f"    Highest:               {max(scores)}/5.0")
            print(f"    Lowest:                {min(scores)}/5.0")

            # Per-chunk final scores
            for er in eval_results:
                r = er.get("rubric_evaluation", {})
                cid = er.get("chunk_id", "?")
                da = r.get("doctrinal_accuracy", {}).get("score", "?")
                tc = r.get("terminology_consistency", {}).get("score", "?")
                print(f"    Chunk {cid}: doctrinal={da}/5, terminology={tc}/5")
    else:
        print("\n  Evaluation skipped (use without --skip-eval to enable)")

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Source:       {doc['filename']} ({source_lang})")
    print(f"  Target:       {args.target}")
    print(f"  Chunks:       {len(chunks)}")
    print(f"  Translations: {translations_path}")
    if not args.skip_eval:
        print(f"  Final:        {final_translations_path}")
        print(f"  Evaluations:  {eval_path}")
    print(f"  Glossary:     {glossary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
