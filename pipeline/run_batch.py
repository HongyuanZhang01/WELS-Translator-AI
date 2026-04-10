"""
run_batch.py - Batch runner for Jonathan's April 2026 test plan

Translates 4 documents into Spanish and Hungarian:
  1. Wauwatosa Theology Paper (English)
  2. Otto J Convention Essay (English)
  3. Augsburg Confession (Latin)
  4. Martin Luther's Large Catechism (German)

USAGE:
  Set your API key first:
    export ANTHROPIC_API_KEY="your-key-here"

  Then run:
    python -m pipeline.run_batch

  Or run a specific document/language combo:
    python -m pipeline.run_batch --doc wauwatosa --lang Spanish
    python -m pipeline.run_batch --doc catechism --lang Hungarian

  Or skip evaluation for faster output (translate only):
    python -m pipeline.run_batch --skip-eval

  Available --doc values: wauwatosa, otto, augsburg, catechism, all
  Available --lang values: Spanish, Hungarian, all
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# Document definitions: (short_name, filename, source_language, context_description)
DOCUMENTS = {
    "wauwatosa": {
        "file": "data/source_docs/wauwatosa_theology_paper.txt",
        "language": "English",
        "context": "Wauwatosa Theology and Its Impact on Pastoral Study, Preaching and Teaching — "
                   "a WELS conference paper on the historical-grammatical approach to Scripture",
    },
    "otto": {
        "file": "data/source_docs/otto_j_convention_essay.txt",
        "language": "English",
        "context": "175 Years of God's Surprising Grace to the Wisconsin Synod — "
                   "a convention essay on WELS history and God's faithfulness",
    },
    "augsburg": {
        "file": "data/source_docs/augsburg_confession_latin.txt",
        "language": "Latin",
        "context": "Confessio Augustana (Augsburg Confession) — the foundational confession "
                   "of Lutheran doctrine presented to Emperor Charles V in 1530, by Philip Melanchthon",
    },
    "catechism": {
        "file": "data/source_docs/large_catechism_german_clean.txt",
        "language": "German",
        "context": "Der große Katechismus (Large Catechism) by Martin Luther — "
                   "a thorough explanation of the Ten Commandments, the Creed, the Lord's Prayer, "
                   "Baptism, and the Lord's Supper, from the Book of Concord (Dresden 1580)",
    },
}

TARGET_LANGUAGES = ["Spanish", "Hungarian"]


def run_job(doc_name, doc_info, target_lang, skip_eval=False, max_loops=3):
    """Run a single translation job using run_pipeline.py."""
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(pipeline_dir, doc_info["file"])

    # Create output directory for this specific job
    output_dir = os.path.join(
        pipeline_dir, "data", "batch_april2026",
        f"{doc_name}_{target_lang.lower()}"
    )
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pipeline.run_pipeline",
        source_file,
        "--target", target_lang,
        "--language", doc_info["language"],
        "--context", doc_info["context"],
        "--output-dir", output_dir,
        "--max-loops", str(max_loops),
    ]

    if skip_eval:
        cmd.append("--skip-eval")

    print(f"\n{'#' * 70}")
    print(f"# JOB: {doc_name} -> {target_lang}")
    print(f"# Source: {doc_info['file']} ({doc_info['language']})")
    print(f"# Output: {output_dir}")
    print(f"{'#' * 70}\n")

    start_time = datetime.now()
    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(pipeline_dir),  # Run from the repo root
    )
    elapsed = datetime.now() - start_time

    status = "SUCCESS" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    print(f"\n  [{doc_name} -> {target_lang}] {status} in {elapsed}")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Batch runner for April 2026 translation test plan")
    parser.add_argument(
        "--doc", type=str, default="all",
        choices=list(DOCUMENTS.keys()) + ["all"],
        help="Which document to translate (default: all)"
    )
    parser.add_argument(
        "--lang", type=str, default="all",
        choices=TARGET_LANGUAGES + ["all"],
        help="Which target language (default: all)"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation (faster, translate only)"
    )
    parser.add_argument(
        "--max-loops", type=int, default=3,
        help="Max improvement loops per chunk (default: 3)"
    )
    args = parser.parse_args()

    # Verify API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set!")
        print("  Run: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    # Determine which jobs to run
    docs = list(DOCUMENTS.keys()) if args.doc == "all" else [args.doc]
    langs = TARGET_LANGUAGES if args.lang == "all" else [args.lang]

    total_jobs = len(docs) * len(langs)
    print(f"=" * 70)
    print(f"WELS Translation Batch Runner — April 2026 Test Plan")
    print(f"  Documents: {', '.join(docs)}")
    print(f"  Languages: {', '.join(langs)}")
    print(f"  Total jobs: {total_jobs}")
    print(f"  Eval: {'SKIPPED' if args.skip_eval else f'enabled (max {args.max_loops} loops)'}")
    print(f"=" * 70)

    # Run jobs in order: smallest documents first for quick feedback
    results = {}
    job_num = 0
    batch_start = datetime.now()

    for doc_name in docs:
        for lang in langs:
            job_num += 1
            print(f"\n{'=' * 70}")
            print(f"  Starting job {job_num}/{total_jobs}")
            print(f"{'=' * 70}")

            success = run_job(
                doc_name, DOCUMENTS[doc_name], lang,
                skip_eval=args.skip_eval,
                max_loops=args.max_loops,
            )
            results[(doc_name, lang)] = success

    # Summary
    batch_elapsed = datetime.now() - batch_start
    print(f"\n\n{'=' * 70}")
    print(f"BATCH COMPLETE — {batch_elapsed}")
    print(f"{'=' * 70}")
    for (doc, lang), success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {doc} -> {lang}")

    failed = sum(1 for v in results.values() if not v)
    if failed:
        print(f"\n  WARNING: {failed} job(s) failed!")
    else:
        print(f"\n  All {total_jobs} jobs completed successfully!")


if __name__ == "__main__":
    main()
