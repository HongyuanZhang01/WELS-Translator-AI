"""
merge_eval_outputs.py - Combine partial + resume evaluation runs into a single set

WHAT THIS IS FOR
================
When an evaluation run crashes midway (say, at chunk 19 of 37), you fix the
crash, then relaunch with `python -m pipeline.run_eval ... --chunk-start 19`.
That resume run produces a SECOND set of output files covering only chunks
19-37, while the original (dead) run's files still cover chunks 1-18.

The packaging program expects exactly ONE clean evaluations file and ONE
clean evaluated_translations file per job folder, so before you can package
a job that had a resume run, you need to merge the partial outputs into a
single coherent pair.

That's what this script does.

WHAT IT DOES
============
    1. Looks inside a job folder (e.g. pipeline/data/batch_april2026/otto_hungarian).
    2. Finds every non-merged evaluations_<target>_*.json file and its
       matching evaluated_translations_<target>_*.json file.
    3. Orders them oldest -> newest by the timestamp embedded in the filename.
    4. For every chunk_id, keeps the version from the NEWEST run that
       contains it. So if the original run died after chunk 18 and the
       resume covered chunks 19-37, you get chunks 1-18 from the original
       and chunks 19-37 from the resume.
    5. Validates that the merged set covers every chunk_id from 001 through
       the total in chunks_*.json with no gaps.
    6. Writes two new files into the same job folder:
           evaluations_<target>_<today>_MERGED.json
           evaluated_translations_<target>_<today>_MERGED.json
       The originals are NEVER touched.

SAFETY
======
    - Originals are never modified or deleted.
    - Refuses to run if any *_MERGED.json already exists, unless you pass
      --force (then the old merged files are simply left alone and the new
      merged files get a fresh timestamp — no overwrites either way).
    - Refuses if the merge would leave chunks missing from the 1..N range.
    - Refuses if the merge would have zero non-merged inputs (nothing to
      merge).

USAGE
=====
    python -m pipeline.merge_eval_outputs pipeline/data/batch_april2026/otto_hungarian
    python -m pipeline.merge_eval_outputs pipeline/data/batch_april2026/otto_spanish --force
"""

import os
import sys
import glob
import json
import argparse
import re
from datetime import datetime


# Pattern pieces used below
# Files look like: evaluations_hungarian_20260410_082252.json
#                  evaluated_translations_hungarian_20260410_082252.json
TIMESTAMP_IN_NAME = re.compile(r"(\d{8}_\d{6})")


def find_exactly_one(folder, pattern, description):
    """Locate exactly one file matching `pattern` in `folder`.
    Abort with a clear message on zero or multiple matches."""
    matches = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(matches) == 0:
        raise SystemExit(
            f"ERROR: no {description} found in {folder} "
            f"(pattern: {pattern}). Cannot proceed."
        )
    if len(matches) > 1:
        raise SystemExit(
            f"ERROR: {len(matches)} {description} files found in {folder}, "
            f"expected exactly 1. Found:\n  "
            + "\n  ".join(os.path.basename(m) for m in matches)
        )
    return matches[0]


def load_json(path, description):
    """Load a JSON file and abort cleanly on parse errors."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"ERROR: {description} at {path} is not valid JSON.\n"
            f"  Parse error: {e}\n"
            f"  This usually means a run crashed mid-write. The file is "
            f"truncated. Try the previous run's output if there is one, "
            f"or re-run eval on this job."
        )


def detect_target_language(folder):
    """Look at the filenames in the folder to figure out which target language
    this job is for (spanish, hungarian, etc.). We use the target_language
    field from the data as the authoritative source later — this is just for
    file discovery."""
    evaluations_files = glob.glob(
        os.path.join(folder, "evaluations_*.json")
    )
    # Filter out *_MERGED.json files — we only look at raw run outputs here.
    evaluations_files = [f for f in evaluations_files if ("_MERGED" not in f and "_FIXED" not in f)]
    if not evaluations_files:
        raise SystemExit(
            f"ERROR: no non-merged evaluations_*.json files found in {folder}. "
            f"There is nothing to merge."
        )
    # Grab the target language token from the first filename.
    # Pattern: evaluations_<target>_<timestamp>.json
    first = os.path.basename(evaluations_files[0])
    parts = first[len("evaluations_"):].rsplit("_", 2)
    # parts looks like: ["hungarian", "20260410", "082252.json"]
    if len(parts) < 3:
        raise SystemExit(
            f"ERROR: cannot parse target language from filename {first}. "
            f"Expected pattern: evaluations_<target>_<YYYYMMDD>_<HHMMSS>.json"
        )
    return parts[0]


def timestamp_from_name(path):
    """Extract the YYYYMMDD_HHMMSS string from a filename. We sort runs by
    this so the NEWEST run wins on any chunk overlap."""
    match = TIMESTAMP_IN_NAME.search(os.path.basename(path))
    if not match:
        raise SystemExit(
            f"ERROR: cannot find a YYYYMMDD_HHMMSS timestamp in "
            f"{os.path.basename(path)}. I don't know how to order this run "
            f"relative to the others."
        )
    return match.group(1)


def merge_run_outputs(run_files, kind):
    """Given a list of (timestamp, path) pairs sorted oldest -> newest,
    load each one and fold its chunks into a dict keyed by chunk_id, with
    the newest run winning on overlap. Returns a list sorted by chunk_id.

    `kind` is "evaluations" or "evaluated_translations" — used only for
    error messages."""
    merged = {}  # chunk_id -> chunk dict
    source_of = {}  # chunk_id -> which timestamp provided it (for the report)
    for timestamp, path in run_files:
        data = load_json(path, f"{kind} file {os.path.basename(path)}")
        if not isinstance(data, list):
            raise SystemExit(
                f"ERROR: {path} has wrong top-level type: expected list, "
                f"got {type(data).__name__}"
            )
        for item in data:
            if not isinstance(item, dict) or "chunk_id" not in item:
                raise SystemExit(
                    f"ERROR: {path} contains an entry without a chunk_id:\n  {item!r}"
                )
            chunk_id = item["chunk_id"]
            merged[chunk_id] = item
            source_of[chunk_id] = timestamp
    # Return sorted by chunk_id so the merged file looks natural.
    ordered_ids = sorted(merged.keys())
    return [merged[cid] for cid in ordered_ids], source_of


def expected_chunk_ids(folder):
    """Read chunks_*.json to learn how many chunks this job has total, and
    return the expected set of chunk_id strings ('001', '002', ...)."""
    chunks_path = find_exactly_one(folder, "chunks_*.json", "chunks file")
    chunks = load_json(chunks_path, "chunks file")
    if not isinstance(chunks, list):
        raise SystemExit(
            f"ERROR: chunks file has wrong top-level type: "
            f"expected list, got {type(chunks).__name__}"
        )
    expected = set()
    for item in chunks:
        if not isinstance(item, dict) or "chunk_id" not in item:
            raise SystemExit(
                f"ERROR: chunks file contains an entry without chunk_id"
            )
        expected.add(item["chunk_id"])
    return expected, len(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Merge partial + resume eval runs into a single clean pair",
    )
    parser.add_argument(
        "job_folder", type=str,
        help="Path to a job folder (e.g. pipeline/data/batch_april2026/otto_hungarian)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Run even if *_MERGED.json files already exist in the folder "
             "(the new merged files get a new timestamp; old ones are left alone).",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.job_folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"ERROR: job folder does not exist: {folder}")

    print("=" * 60)
    print("merge_eval_outputs.py")
    print(f"Job folder: {folder}")
    print("=" * 60)

    target = detect_target_language(folder)
    print(f"\nDetected target language: {target}")

    # Pre-flight: are there existing merged files?
    existing_merged = sorted(glob.glob(os.path.join(folder, "*_MERGED.json")))
    if existing_merged and not args.force:
        raise SystemExit(
            "\nERROR: this folder already has merged files:\n  "
            + "\n  ".join(os.path.basename(m) for m in existing_merged)
            + "\n\nRefusing to run without --force. The old merged files will\n"
              "NOT be deleted if you pass --force — new merged files just get\n"
              "a fresh timestamp so both versions live side by side."
        )

    # Gather every non-merged evaluations_<target>_*.json + its matching
    # evaluated_translations_<target>_*.json partner.
    all_eval_files = sorted(glob.glob(
        os.path.join(folder, f"evaluations_{target}_*.json")
    ))
    all_eval_files = [f for f in all_eval_files if ("_MERGED" not in f and "_FIXED" not in f)]

    all_etrans_files = sorted(glob.glob(
        os.path.join(folder, f"evaluated_translations_{target}_*.json")
    ))
    all_etrans_files = [f for f in all_etrans_files if ("_MERGED" not in f and "_FIXED" not in f)]

    if not all_eval_files:
        raise SystemExit(
            f"\nERROR: no evaluations_{target}_*.json files to merge in {folder}."
        )
    if not all_etrans_files:
        raise SystemExit(
            f"\nERROR: no evaluated_translations_{target}_*.json files to merge in {folder}."
        )

    # Pair them by timestamp. A pair is (timestamp, eval_path, etrans_path).
    pairs = []
    etrans_by_ts = {timestamp_from_name(f): f for f in all_etrans_files}
    for ef in all_eval_files:
        ts = timestamp_from_name(ef)
        if ts not in etrans_by_ts:
            raise SystemExit(
                f"\nERROR: evaluations file {os.path.basename(ef)} has no "
                f"matching evaluated_translations_{target}_{ts}.json. "
                f"Cannot merge a half-paired run."
            )
        pairs.append((ts, ef, etrans_by_ts[ts]))

    # Sort pairs oldest -> newest so newer runs win on overlap.
    pairs.sort(key=lambda p: p[0])

    print(f"\nFound {len(pairs)} run(s) to merge (oldest first):")
    for ts, ef, etf in pairs:
        print(f"  {ts}")
        print(f"    {os.path.basename(ef)}")
        print(f"    {os.path.basename(etf)}")

    if len(pairs) == 1:
        print("\nNOTE: only one run found. There is nothing to merge — the")
        print("job folder already has a single coherent run. You probably")
        print("don't need this tool for this job.")
        # Still proceed, so the user gets merged files if they really want them.

    # Merge.
    print("\nMerging evaluations files...")
    eval_runs = [(ts, ef) for (ts, ef, _) in pairs]
    merged_evals, eval_source = merge_run_outputs(eval_runs, "evaluations")

    print("Merging evaluated_translations files...")
    etrans_runs = [(ts, etf) for (ts, _, etf) in pairs]
    merged_etrans, etrans_source = merge_run_outputs(
        etrans_runs, "evaluated_translations"
    )

    # Validate coverage against chunks_*.json.
    expected_ids, total_chunks = expected_chunk_ids(folder)
    merged_eval_ids = {c["chunk_id"] for c in merged_evals}
    merged_etrans_ids = {c["chunk_id"] for c in merged_etrans}

    missing_eval = sorted(expected_ids - merged_eval_ids)
    missing_etrans = sorted(expected_ids - merged_etrans_ids)
    if missing_eval or missing_etrans:
        raise SystemExit(
            f"\nERROR: merge would leave chunks missing.\n"
            f"  chunks_*.json has {total_chunks} total chunks.\n"
            f"  evaluations coverage missing: {missing_eval or 'none'}\n"
            f"  evaluated_translations coverage missing: {missing_etrans or 'none'}\n"
            f"\nTo fix this, resume the eval run for the missing chunk range, "
            f"then re-run this merger."
        )

    extra_eval = sorted(merged_eval_ids - expected_ids)
    extra_etrans = sorted(merged_etrans_ids - expected_ids)
    if extra_eval or extra_etrans:
        print(
            f"\nWARNING: merge contains chunks not present in chunks_*.json\n"
            f"  evaluations extras: {extra_eval or 'none'}\n"
            f"  evaluated_translations extras: {extra_etrans or 'none'}"
        )
        print("  Proceeding anyway — these chunks will be kept in the merged output.")

    # Report which run contributed each chunk — very handy for debugging
    # "wait, chunk 25 looks different than I expected."
    print("\nChunk-by-chunk provenance (which run contributed each chunk):")
    provenance = {}
    for cid in sorted(merged_eval_ids):
        ts = eval_source[cid]
        provenance.setdefault(ts, []).append(cid)
    for ts, cids in sorted(provenance.items()):
        print(f"  from run {ts}: {len(cids)} chunks")
        # Show compact ranges: 1-18, 19-37
        nums = sorted(int(c) for c in cids)
        ranges = []
        start = nums[0]
        prev = nums[0]
        for n in nums[1:]:
            if n == prev + 1:
                prev = n
            else:
                ranges.append(f"{start:03d}-{prev:03d}" if start != prev else f"{start:03d}")
                start = n
                prev = n
        ranges.append(f"{start:03d}-{prev:03d}" if start != prev else f"{start:03d}")
        print(f"    ranges: {', '.join(ranges)}")

    # Write the merged files.
    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_eval_path = os.path.join(
        folder, f"evaluations_{target}_{today}_MERGED.json"
    )
    merged_etrans_path = os.path.join(
        folder, f"evaluated_translations_{target}_{today}_MERGED.json"
    )

    print(f"\nWriting {os.path.basename(merged_eval_path)}...")
    with open(merged_eval_path, "w", encoding="utf-8") as f:
        json.dump(merged_evals, f, ensure_ascii=False, indent=2)

    print(f"Writing {os.path.basename(merged_etrans_path)}...")
    with open(merged_etrans_path, "w", encoding="utf-8") as f:
        json.dump(merged_etrans, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"  Merged {len(pairs)} run(s) into {total_chunks} chunks.")
    print(f"  Merged files written with timestamp {today}.")
    print("  Original run files were NOT modified.")


if __name__ == "__main__":
    main()
