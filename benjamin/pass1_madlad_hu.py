"""
translate_nllb_hu.py
NLLB-based German → Hungarian translation over Walther chunks.

Input:
  - chunks.json          : JSON array of {chunk_id, lecture, thesis, text}

Output:
  - ~/ge_rig/output/hu_nllb/pass1/<chunk_id>.txt

Usage:
  python translate_nllb_hu.py [start] [end] [max_workers]
  python translate_nllb_hu.py 1 50 8
"""

import sys
import os
import json
import concurrent.futures
import requests
import re
import time
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR    = os.path.expanduser("~/ge_rig")
CHUNKS_FILE = os.path.join(BASE_DIR, "output", "chunks.json")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output", "hu_madlad")

API_URL     = "http://localhost:8001/translate_batch"
SRC_LANG    = "de"
TGT_LANG    = "hu"
BATCH_SIZE  = 256    # Sentences per API call — reduced to limit degeneration risk
MAX_WORKERS = 8     # Parallel chunk threads

# Soft cap: sentences longer than this (in characters) get flagged in output
# but are still sent — hard splitting mid-sentence does more harm than good
LONG_SENTENCE_WARN = 600

# =============================================================================
# SENTENCE SPLITTING
# =============================================================================

def split_sentences(text: str) -> list[str]:

    protected = re.sub(r'([.!?])([A-ZÁÉÍÓÖŐÚÜŰ])', r'\1 \2', text)    

    # Protect parenthetical expressions like (12. September 1884.)
    protected = re.sub(r'\([^)]+\)', lambda m: m.group(0).replace('.', '<<DOT>>'), protected)
    
    # Protect numbered list items like "1." "2." at start of clause
    protected = re.sub(r'(\s)(\d+\.)(\s)(?=\w)', r'\1\2<<SPACE>>', protected)
    
    # Protect common German abbreviations
    protected = re.sub(
        r'\b(Dr|Mr|St|Str|bzw|ca|etc|usw|vgl|z\.B|d\.h|u\.a|Jhd|Jh|ff|ꝛc)\.',
        lambda m: m.group(0).replace('.', '<<DOT>>'),
        protected
    )

    # Protect punctuation inside quoted speech from triggering splits
    protected = re.sub(
        r'(„[^"]+?")',
        lambda m: m.group(0).replace('?', '<<QMARK>>').replace('!', '<<EMARK>>').replace('.', '<<DOT>>'),
        protected
    )

    # Normalize missing spaces before citations like "selig.Luc."
    protected = re.sub(r'([.!?])([A-Z][a-z]{0,3}\.)', r'\1 \2', protected)

    # Split on sentence-ending punctuation before uppercase
    # OR lowercase-initial subject-dropped sentences
    # OR after closing quote before uppercase
    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰ\"\„])'
        r'|(?<=\.)\s+(?=[a-z])'
        r'|(?<=[\u0022\u201C\u201D\u00AB\u00BB])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰ])'
        r'|(?<=[.!?"])\s*—\s*(?=[A-ZÁÉÍÓÖŐÚÜŰ])',
        protected
    )

    # Restore all placeholders
    sentences = [
        s.replace('<<DOT>>', '.')
         .replace('<<SPACE>>', ' ')
         .replace('<<QMARK>>', '?')
         .replace('<<EMARK>>', '!')
        for s in sentences
    ]

    # Prefix lowercase-starting sentences with explicit subject
    rejoined = []
    for s in sentences:
        if not s:
            continue
        if s[0].islower():
            rejoined.append('Es ' + s)
        else:
            rejoined.append(s)

    # Second pass: split long sentences on semicolons
    result = []
    for s in rejoined:
        if len(s) > LONG_SENTENCE_WARN and ';' in s:
            parts = re.split(r';\s*', s)
            result.extend(parts)
        else:
            result.append(s)

    return [s.strip() for s in result if s.strip()]


# =============================================================================
# TRANSLATION
# =============================================================================

def translate_chunk(chunk: dict) -> tuple:
    """
    Translate a single chunk via NLLB. Splits the text into sentences,
    batches them for the API, and writes the result to pass1/<chunk_id>.txt.

    Output preserves one translated sentence per line, so downstream
    models can see the sentence-level structure.

    Returns (chunk_id, error_string_or_None).
    """
    chunk_id = str(chunk["chunk_id"])
    out_path = os.path.join(OUTPUT_DIR, "pass1", f"{chunk_id}.txt")

    if os.path.exists(out_path):
        return chunk_id, None  # already done

    text = chunk["text"].strip()
    if not text:
        return chunk_id, "Empty chunk text — skipped."

    sentences = split_sentences(text)

    # Temporary diagnostic — remove after verification
    for i, s in enumerate(sentences):
        print(f"  [{chunk_id}] sentence {i+1}: {s[:120]}")

    # Warn on any sentences that survived splitting but are still very long
    for i, s in enumerate(sentences):
        if len(s) > LONG_SENTENCE_WARN:
            tqdm.write(f"  [{chunk_id}] Long sentence #{i+1} ({len(s)} chars) — "
                       f"degeneration risk: {s[:80]}...")

    translated_lines = []
    try:
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i : i + BATCH_SIZE]
            response = requests.post(
                API_URL,
                json={"sentences": batch, "src_lang": SRC_LANG, "tgt_lang": TGT_LANG},
                timeout=600,
            )
            response.raise_for_status()
            translated_lines.extend(response.json()["translations"])
    except Exception as e:
        return chunk_id, str(e)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(translated_lines))

    return chunk_id, None


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(start_idx: int, end_idx: int, max_workers: int):

    t0 = time.time()
    os.makedirs(os.path.join(OUTPUT_DIR, "pass1"), exist_ok=True)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    all_chunks.sort(key=lambda c: int(str(c["chunk_id"])))
    target_chunks = all_chunks[start_idx - 1 : end_idx]

    print(f"NLLB translation | language: Hungarian | chunks {start_idx}–{end_idx} "
          f"({len(target_chunks)} chunks) | workers: {max_workers}")
    print(f"Output: {OUTPUT_DIR}\n")

    pending, skipped = [], 0
    for chunk in target_chunks:
        cid = str(chunk["chunk_id"])
        if os.path.exists(os.path.join(OUTPUT_DIR, "pass1", f"{cid}.txt")):
            skipped += 1
        else:
            pending.append(chunk)

    if skipped:
        print(f"Skipping {skipped} already-completed chunks.")
    print(f"Processing {len(pending)} chunks.\n")

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(translate_chunk, c): c["chunk_id"] for c in pending}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc="NLLB pass1"):
            chunk_id, error = future.result()
            if error:
                errors.append((chunk_id, error))
                tqdm.write(f"  [{chunk_id}] ERROR: {error}")

    print(f"\n=== SUMMARY ===")
    print(f"Completed: {len(pending) - len(errors) + skipped}/{len(target_chunks)} "
          f"({skipped} pre-existing, {len(errors)} errors)")

    print(f"[{chunk_id}] {time.time()-t0:.1f}s")

    if errors:
        print("Errors:")
        for cid, err in errors:
            print(f"  [{cid}] {err}")


if __name__ == "__main__":
    start   = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end     = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_WORKERS
    run(start, end, workers)