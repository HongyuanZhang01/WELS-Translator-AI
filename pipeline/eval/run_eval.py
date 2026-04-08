"""
run_eval.py - Evaluate ANY translation, from ANY source

You use this when you ALREADY HAVE a translation and just want to grade it.
(The pipeline runs the evaluator automatically — this is for everything else.)

SIMPLE COMMANDS:

  Grade our pipeline's translations:
    python -m pipeline.eval.run_eval -p pipeline/data/translations_hungarian_20260406.json

  Grade Benjamin's translations:
    python -m pipeline.eval.run_eval -b pass2 -c 012
    python -m pipeline.eval.run_eval -b pass2 --all

  Grade any two text files:
    python -m pipeline.eval.run_eval -s original.txt -t translated.txt --lang Hungarian

  Compare Benjamin's pass2 vs final_differ:
    python -m pipeline.eval.run_eval --compare 012

  See what's available to evaluate:
    python -m pipeline.eval.run_eval --list

  Then turn results into a spreadsheet:
    python -m pipeline.eval.export_to_excel

INPUT:  Source text + translation (from any of the above sources)
OUTPUT: JSON files in pipeline/eval/results/ + terminal summary
        Then export_to_excel turns those JSONs into a 5-sheet Excel report.
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CHUNKS_FILE,
    PASS2_DIR,
    FINAL_DIFFER_DIR,
    EVAL_OUTPUT_DIR,
    DEFAULT_TARGET,
)
from eval.evaluator import full_evaluation


# =========================================================================
# DATA LOADERS — handle all the different input formats
# =========================================================================

def load_benjamin_chunks():
    """Load Benjamin's source chunks from chunks.json."""
    chunks_path = os.path.abspath(CHUNKS_FILE)
    if not os.path.exists(chunks_path):
        print(f"  WARNING: Benjamin's chunks.json not found at {chunks_path}")
        return {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {chunk["chunk_id"]: chunk for chunk in chunks}


def load_benjamin_translation(chunk_id, source_dir):
    """Load one of Benjamin's translation files (pass2 or final_differ)."""
    filepath = os.path.join(os.path.abspath(source_dir), f"{chunk_id}.txt")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    # Strip Benjamin's DEBUG glossary sections
    if "DEBUG" in text and "GLOSSARY TERMS" in text:
        text = text.split("=" * 60)[0].strip()
    return text


def load_benjamin_glossary():
    """Load Benjamin's glossary and convert to {source: target} dict."""
    glossary_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "glossary_hu.json"
    )
    glossary_path = os.path.abspath(glossary_path)
    if not os.path.exists(glossary_path):
        return None
    with open(glossary_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    glossary = {}
    for entry in raw:
        term = entry.get("term", "")
        hungarian = entry.get("hungarian", "")
        if term and hungarian:
            glossary[term] = hungarian
    return glossary


def load_pipeline_output(filepath):
    """Load translations from our pipeline's JSON output."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pipeline_glossary(data_dir, target_language):
    """Load the pipeline-generated glossary for a target language."""
    glossary_path = os.path.join(data_dir, f"glossary_{target_language.lower()}.json")
    if not os.path.exists(glossary_path):
        return None
    with open(glossary_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    glossary = {}
    for term, info in raw.items():
        if isinstance(info, dict):
            glossary[term] = info.get("translation", "")
        else:
            glossary[term] = str(info)
    return glossary


def get_available_benjamin_chunks():
    """Find which chunk IDs have translations in pass2 and/or final_differ."""
    pass2_path = os.path.abspath(PASS2_DIR)
    differ_path = os.path.abspath(FINAL_DIFFER_DIR)
    pass2_ids = set()
    differ_ids = set()
    if os.path.exists(pass2_path):
        pass2_ids = {f.replace(".txt", "") for f in os.listdir(pass2_path) if f.endswith(".txt")}
    if os.path.exists(differ_path):
        differ_ids = {f.replace(".txt", "") for f in os.listdir(differ_path) if f.endswith(".txt")}
    return {
        "pass2": sorted(pass2_ids),
        "final_differ": sorted(differ_ids),
        "both": sorted(pass2_ids & differ_ids),
    }


# =========================================================================
# RESULT SAVING & PRINTING
# =========================================================================

def save_result(result, label):
    """Save an evaluation result to a timestamped JSON file."""
    output_dir = os.path.abspath(EVAL_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_id = result.get("chunk_id", "unknown")
    filename = f"eval_{label}_{chunk_id}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filepath}")
    return filepath


def print_summary(result):
    """Print a human-readable summary in the terminal."""
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY — Chunk {result.get('chunk_id', '?')}")
    print(f"Source: {result.get('source_label', '?')}")
    print(f"Target Language: {result.get('target_language', '?')}")
    print("=" * 60)

    rubric = result.get("rubric_evaluation", {})
    if "error" in rubric:
        print(f"\n  ERROR: {rubric['error']}")
        return

    print("\n  RUBRIC SCORES:")
    for dim in ["doctrinal_accuracy", "terminology_consistency", "clarity", "naturalness"]:
        if dim in rubric:
            score = rubric[dim].get("score", "?")
            print(f"    {dim}: {score}/5")

    if "weighted_score" in rubric:
        print(f"\n  WEIGHTED OVERALL: {rubric['weighted_score']}/5.0")

    consistency = rubric.get("consistency_report", {})
    if consistency:
        flags = [dim for dim, info in consistency.items() if not info.get("agreed", True)]
        if flags:
            print(f"\n  CONSISTENCY FLAGS: {', '.join(flags)}")
            for dim in flags:
                info = consistency[dim]
                print(f"    {dim}: scores were {info['all_scores']} -> final {info['final_score']}")
        else:
            print("\n  CONSISTENCY: All dimensions agreed across both passes.")

    errors = rubric.get("critical_errors", [])
    if errors:
        print(f"\n  CRITICAL ERRORS ({len(errors)}):")
        for err in errors[:5]:
            print(f"    - {err[:120]}")
    else:
        print("\n  No critical errors found.")

    bt = result.get("back_translation", {}).get("comparison", {})
    if "error" not in bt:
        preserved = bt.get("meaning_preserved", "?")
        print(f"\n  BACK-TRANSLATION: Meaning preserved = {preserved}")
        diffs = bt.get("differences", [])
        critical_diffs = [d for d in diffs if d.get("severity") == "critical"]
        if critical_diffs:
            print(f"  CRITICAL MEANING SHIFTS: {len(critical_diffs)}")
            for d in critical_diffs:
                print(f"    - {d.get('explanation', '')[:120]}")
        elif diffs:
            print(f"  Differences found: {len(diffs)} (none critical)")

    print("\n" + "=" * 60)


# =========================================================================
# EVALUATION RUNNERS
# =========================================================================

def eval_benjamin(chunk_id, source_dir, label, chunks, glossary=None):
    """Evaluate one of Benjamin's translations."""
    if chunk_id not in chunks:
        print(f"  ERROR: Chunk {chunk_id} not found in chunks.json")
        return None
    source_text = chunks[chunk_id]["text"]
    translated_text = load_benjamin_translation(chunk_id, source_dir)
    if translated_text is None:
        print(f"  ERROR: No translation found for chunk {chunk_id} in {source_dir}")
        return None

    print(f"\nEvaluating chunk {chunk_id} ({label})...")
    result = full_evaluation(
        source_text=source_text,
        translated_text=translated_text,
        target_language=DEFAULT_TARGET,
        chunk_id=chunk_id,
        glossary=glossary,
        source_label=f"benjamin_{label}",
    )
    save_result(result, f"benjamin_{label}")
    print_summary(result)
    return result


def eval_pipeline_output(filepath):
    """Evaluate translations from our pipeline's output file."""
    data = load_pipeline_output(filepath)
    data_dir = os.path.dirname(os.path.abspath(filepath))
    results = []
    for i, entry in enumerate(data):
        chunk_id = entry.get("chunk_id", f"{i+1:03d}")
        source_text = entry.get("source_text", "")
        translated_text = entry.get("translated_text", "")
        target_language = entry.get("target_language", DEFAULT_TARGET)
        if not source_text or not translated_text:
            print(f"  Skipping chunk {chunk_id}: missing source or translation")
            continue
        glossary = entry.get("glossary_used", None)
        if not glossary:
            glossary = load_pipeline_glossary(data_dir, target_language)

        print(f"\nEvaluating chunk {chunk_id} ({i+1}/{len(data)})...")
        result = full_evaluation(
            source_text=source_text,
            translated_text=translated_text,
            target_language=target_language,
            chunk_id=chunk_id,
            glossary=glossary,
            source_label="pipeline",
        )
        save_result(result, "pipeline")
        print_summary(result)
        results.append(result)
    return results


def eval_direct(source_file, translation_file, target_language, glossary_file=None):
    """Evaluate any two text files directly."""
    with open(source_file, "r", encoding="utf-8") as f:
        source_text = f.read()
    with open(translation_file, "r", encoding="utf-8") as f:
        translated_text = f.read()

    glossary = None
    if glossary_file:
        with open(glossary_file, "r", encoding="utf-8") as f:
            glossary = json.load(f)
        if isinstance(glossary, list):
            glossary = {e.get("term", ""): e.get(target_language.lower(), "") for e in glossary}

    source_name = os.path.basename(source_file)
    trans_name = os.path.basename(translation_file)

    print(f"\nEvaluating: {source_name} -> {trans_name} ({target_language})")
    result = full_evaluation(
        source_text=source_text,
        translated_text=translated_text,
        target_language=target_language,
        chunk_id=f"{source_name}_vs_{trans_name}",
        glossary=glossary,
        source_label="direct",
    )
    save_result(result, "direct")
    print_summary(result)
    return result


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ANY translation from ANY source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python -m pipeline.eval.run_eval -p pipeline/data/translations_hungarian_*.json
  python -m pipeline.eval.run_eval -b pass2 -c 012
  python -m pipeline.eval.run_eval -b pass2 --all
  python -m pipeline.eval.run_eval -s original.txt -t translated.txt --lang Hungarian
  python -m pipeline.eval.run_eval --compare 012
  python -m pipeline.eval.run_eval --list
        """
    )

    # What to evaluate (pick one)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "-p", "--pipeline", type=str, metavar="FILE",
        help="Grade our pipeline's translation output JSON"
    )
    source_group.add_argument(
        "-b", "--benjamin", type=str, choices=["pass2", "final_differ"],
        help="Grade Benjamin's translations"
    )
    source_group.add_argument(
        "-s", "--source", type=str, metavar="FILE",
        help="Source text file (use with -t)"
    )
    source_group.add_argument(
        "--compare", type=str, metavar="CHUNK_ID",
        help="Compare pass2 vs final_differ for a chunk"
    )
    source_group.add_argument(
        "--list", action="store_true",
        help="Show what's available to evaluate"
    )

    # Options
    parser.add_argument("-t", "--translation", type=str, metavar="FILE",
                        help="Translation file (use with -s)")
    parser.add_argument("-c", "--chunk", type=str, metavar="ID",
                        help="Specific chunk ID")
    parser.add_argument("--lang", type=str, default=DEFAULT_TARGET,
                        help=f"Target language (default: {DEFAULT_TARGET})")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate ALL available chunks")
    parser.add_argument("--glossary", type=str, metavar="FILE",
                        help="Glossary JSON (for -s mode)")

    args = parser.parse_args()

    # --- LIST ---
    if args.list:
        available = get_available_benjamin_chunks()
        print("Benjamin's translations:")
        print(f"  pass2:        {len(available['pass2'])} chunks")
        print(f"  final_differ: {len(available['final_differ'])} chunks")
        print(f"  In both:      {len(available['both'])} chunks")

        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        if os.path.exists(data_dir):
            import glob
            pipeline_files = glob.glob(os.path.join(data_dir, "translations_*.json"))
            if pipeline_files:
                print(f"\nPipeline outputs:")
                for pf in pipeline_files:
                    print(f"  {os.path.basename(pf)}")
        return

    # --- PIPELINE ---
    if args.pipeline:
        eval_pipeline_output(args.pipeline)
        return

    # --- DIRECT FILES ---
    if args.source:
        if not args.translation:
            print("ERROR: -s requires -t (source needs a translation to evaluate)")
            return
        eval_direct(args.source, args.translation, args.lang, args.glossary)
        return

    # --- COMPARE ---
    if args.compare:
        chunks = load_benjamin_chunks()
        glossary = load_benjamin_glossary()
        available = get_available_benjamin_chunks()
        chunk_id = args.compare
        if chunk_id not in available["both"]:
            print(f"ERROR: Chunk {chunk_id} not in both pass2 and final_differ.")
            return

        print(f"\n{'='*60}")
        print(f"COMPARING PIPELINES FOR CHUNK {chunk_id}")
        print(f"{'='*60}")

        r1 = eval_benjamin(chunk_id, os.path.abspath(PASS2_DIR), "pass2", chunks, glossary)
        r2 = eval_benjamin(chunk_id, os.path.abspath(FINAL_DIFFER_DIR), "final_differ", chunks, glossary)

        if r1 and r2:
            s1 = r1.get("rubric_evaluation", {}).get("weighted_score", 0)
            s2 = r2.get("rubric_evaluation", {}).get("weighted_score", 0)
            print(f"\n{'='*60}")
            print(f"  pass2:        {s1}/5.0")
            print(f"  final_differ: {s2}/5.0")
            winner = "pass2" if s1 > s2 else "final_differ" if s2 > s1 else "tie"
            print(f"  Winner: {winner}")
            print(f"{'='*60}")
        return

    # --- BENJAMIN ---
    if args.benjamin:
        chunks = load_benjamin_chunks()
        glossary = load_benjamin_glossary()
        source_dir = os.path.abspath(PASS2_DIR if args.benjamin == "pass2" else FINAL_DIFFER_DIR)
        available = get_available_benjamin_chunks()

        if args.all:
            chunk_ids = available[args.benjamin]
            print(f"\nEvaluating all {len(chunk_ids)} chunks from {args.benjamin}...")
            results = []
            for i, cid in enumerate(chunk_ids):
                print(f"\n--- Chunk {i+1}/{len(chunk_ids)} ---")
                r = eval_benjamin(cid, source_dir, args.benjamin, chunks, glossary)
                if r:
                    results.append(r)
            if results:
                scores = [r["rubric_evaluation"].get("weighted_score", 0)
                          for r in results if "error" not in r.get("rubric_evaluation", {})]
                if scores:
                    print(f"\n{'='*60}")
                    print(f"OVERALL ({args.benjamin}): avg={sum(scores)/len(scores):.2f}, "
                          f"min={min(scores)}, max={max(scores)}, n={len(scores)}")
                    print(f"{'='*60}")
        else:
            chunk_id = args.chunk
            if not chunk_id:
                chunk_id = available[args.benjamin][0] if available[args.benjamin] else None
                if not chunk_id:
                    print("ERROR: No translations available.")
                    return
                print(f"  No chunk specified, using: {chunk_id}")
            eval_benjamin(chunk_id, source_dir, args.benjamin, chunks, glossary)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
