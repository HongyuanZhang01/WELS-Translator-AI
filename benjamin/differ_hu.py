"""
differ_hu.py
Differ arbitrator — compares Qwen Pass 2 and TranslateGemma Hungarian outputs
and produces a finalised Hungarian text for Walther's *The Proper Distinction
Between Law and Gospel*.

Architecture:
  - Qwen Pass 2 output    : semantically corrected, glossary-enforced
  - TranslateGemma output : independent translation, no MadLAD in context
  - Arbitrator model      : receives both candidates + German source + glossary
                            and produces the finalised output

The arbitrator is instructed to:
  - Prefer whichever candidate is more accurate against the German source
  - Enforce glossary terms where either candidate deviates
  - Preserve Latin, Greek, Hebrew exactly
  - Not make generative changes beyond resolving genuine divergences

Input:
  - chunks.json                  : JSON array of {chunk_id, lecture, thesis, text}
  - hu_qwen/pass2/<id>.txt       : Qwen Pass 2 output
  - hu_gemma/translate/<id>.txt  : TranslateGemma independent output
  - hu_gemma/differ/<prev>.txt   : previous finalised chunk (tail only)
  - theological_vocabulary_deduped.json : glossary

Output:
  - hu_gemma/differ/<id>.txt     : finalised Hungarian

Usage:
  python differ_hu.py [start] [end] [max_workers]
  python differ_hu.py 1 50 3
"""

import sys
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR      = os.path.expanduser("~/ge_rig")
CHUNKS_FILE   = os.path.join(BASE_DIR, "output", "chunks.json")
PASS2_DIR     = os.path.join(BASE_DIR, "output", "hu_qwen", "pass2")
GEMMA_DIR     = os.path.join(BASE_DIR, "output", "hu_gemma", "translate")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output", "hu_gemma", "differ")
GLOSSARY_PATH = os.path.join(BASE_DIR, "output", "theological_vocabulary_deduped.json")

# Arbitrator model — swap as needed for testing
MODEL  = "RedHatAI/gemma-3-27b-it-FP8-dynamic"
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-not-needed")

# Characters of previous finalised output to include for style continuity
PREV_TAIL_CHARS = 300

# Separator used in Pass 2 debug output
PASS2_GLOSSARY_SEP = "=" * 60

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """\
You are arbitrating between two independent Hungarian translations of a passage
from C. F. W. Walther's *The Proper Distinction Between Law and Gospel*
(19th-century Lutheran German lectures, 1884–1885).

You will receive:
1. GERMAN SOURCE — the authoritative reference.
2. CANDIDATE A — produced by a semantic correction pipeline (Qwen).
3. CANDIDATE B — produced by an independent translation model (TranslateGemma).
4. GLOSSARY — theological term renderings that are binding.
5. PREVIOUS CHUNK TAIL — the end of the preceding finalised chunk for register
   continuity.

YOUR TASK:
For each part of the text, select the better candidate or, where neither is
fully correct, produce the minimum correction needed. Follow these priorities:

1. ACCURACY — the output must faithfully represent the German source in every
   sentence. Where candidates differ, check against the German and prefer the
   accurate one.

2. GLOSSARY — where either candidate uses a term that conflicts with the
   glossary, correct it to the glossary rendering regardless of which candidate
   it came from.

3. LATIN, GREEK, HEBREW — preserve all non-Hungarian terms exactly as they
   appear in the German source. If either candidate has translated a Latin term
   into Hungarian, restore the original Latin with a bracketed Hungarian gloss:
     subjectum operationis (cselekvés tárgya)
     finis cui (célja)
     in causa formali (a formai ok tekintetében)
     formali causa (formai ok)

4. REGISTER — prefer whichever candidate better matches the register of the
   PREVIOUS CHUNK TAIL. Walther's register is elevated 19th-century Lutheran
   lecture style; biblical quotations follow Károli Bible phrasing.

5. CONSERVATISM — do not make changes beyond resolving genuine divergences.
   If both candidates agree on a passage, reproduce it as-is even if you would
   phrase it differently. Your role is arbitration, not rewriting.

Output only the finalised Hungarian text. No commentary, no headers, no
explanations. Preserve paragraph breaks.\
"""

# =============================================================================
# GLOSSARY LOADING
# Reuse Pass 2 debug block if present; otherwise load from file directly.
# =============================================================================

_GLOSSARY_CACHE = None
_GLOSSARY_LOCK  = threading.Lock()

def _load_raw_glossary() -> list:
    global _GLOSSARY_CACHE
    with _GLOSSARY_LOCK:
        if _GLOSSARY_CACHE is None:
            if os.path.exists(GLOSSARY_PATH):
                with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
                    _GLOSSARY_CACHE = json.load(f)
            else:
                _GLOSSARY_CACHE = []
    return _GLOSSARY_CACHE


def parse_pass2_file(raw: str) -> tuple[str, str]:
    """
    Split Pass 2 output into (translation_body, debug_glossary_block).
    The debug block follows the first ===...=== separator line.
    """
    lines       = raw.split("\n")
    sep_indices = [i for i, l in enumerate(lines)
                   if l.strip() == PASS2_GLOSSARY_SEP]

    if sep_indices:
        body  = "\n".join(lines[:sep_indices[0]]).strip()
        gloss = "\n".join(lines[sep_indices[0] + 1:]).strip()
        return body, gloss

    return raw.strip(), ""


def get_glossary_for_chunk(pass2_gloss_block: str) -> str:
    """
    Use the Pass 2 debug glossary block if available (already chunk-specific).
    Fall back to the full glossary file (first 40 entries) if not.
    """
    if pass2_gloss_block.strip():
        return pass2_gloss_block.strip()

    glossary = _load_raw_glossary()
    lines = []
    for entry in glossary[:40]:
        if not isinstance(entry, dict):
            continue
        term = entry.get("term", "")
        hu   = entry.get("hungarian", "")
        if term and hu:
            lines.append(f"{term} → {hu}")
    return "\n".join(lines)


# =============================================================================
# MODEL CALL
# =============================================================================

def _call(messages: list, timeout: int = 600) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        timeout=timeout,
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# DIFFER LOGIC
# =============================================================================

def arbitrate_chunk(chunk: dict, prev_chunk_id: str | None) -> tuple:
    """
    Arbitrate between Qwen Pass 2 and TranslateGemma for a single chunk.
    Returns (chunk_id, elapsed_or_None, error_or_None).
    """
    chunk_id   = str(chunk["chunk_id"])
    chunk_text = chunk["text"].strip()

    out_path = os.path.join(OUTPUT_DIR, f"{chunk_id}.txt")
    if os.path.exists(out_path):
        return chunk_id, None, None

    # Load Qwen Pass 2 output
    pass2_path = os.path.join(PASS2_DIR, f"{chunk_id}.txt")
    if not os.path.exists(pass2_path):
        return chunk_id, None, f"Pass 2 output missing: {chunk_id}"

    with open(pass2_path, "r", encoding="utf-8") as f:
        pass2_raw = f.read()

    candidate_a, gloss_block = parse_pass2_file(pass2_raw)

    # Load TranslateGemma output
    gemma_path = os.path.join(GEMMA_DIR, f"{chunk_id}.txt")
    if not os.path.exists(gemma_path):
        return chunk_id, None, f"TranslateGemma output missing: {chunk_id}"

    with open(gemma_path, "r", encoding="utf-8") as f:
        candidate_b = f.read().strip()

    # Previous finalised chunk tail for register continuity
    prev_tail = ""
    if prev_chunk_id is not None:
        prev_path = os.path.join(OUTPUT_DIR, f"{prev_chunk_id}.txt")
        if os.path.exists(prev_path):
            with open(prev_path, "r", encoding="utf-8") as f:
                prev_text = f.read().strip()
            if prev_text:
                prev_tail = prev_text[-PREV_TAIL_CHARS:]

    glossary = get_glossary_for_chunk(gloss_block)

    # Assemble user content
    user_content = f"GERMAN SOURCE:\n{chunk_text}\n\n"

    if glossary:
        user_content += f"GLOSSARY:\n{glossary}\n\n"

    user_content += f"CANDIDATE A (Qwen Pass 2):\n{candidate_a}\n\n"
    user_content += f"CANDIDATE B (TranslateGemma):\n{candidate_b}\n\n"

    if prev_tail:
        user_content += (
            f"PREVIOUS CHUNK TAIL (finalised — match this register):\n"
            f"{prev_tail}\n\n"
        )

    user_content += "Produce the finalised Hungarian text now."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    try:
        t0      = time.time()
        result  = _call(messages, timeout=600)
        elapsed = time.time() - t0

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)

        return chunk_id, elapsed, None
    except Exception as e:
        return chunk_id, None, f"Differ error: {e}"


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(start_idx: int, end_idx: int, max_workers: int):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    all_chunks.sort(key=lambda c: int(str(c["chunk_id"])))
    target_chunks = all_chunks[start_idx - 1 : end_idx]

    full_sorted = sorted(all_chunks, key=lambda c: int(str(c["chunk_id"])))
    chunk_index = {str(c["chunk_id"]): i for i, c in enumerate(full_sorted)}

    print(f"Differ arbitrator | Hungarian | chunks {start_idx}–{end_idx} "
          f"({len(target_chunks)} chunks) | workers: {max_workers}")
    print(f"Model:  {MODEL}")
    print(f"Output: {OUTPUT_DIR}\n")

    pending, skipped = [], 0
    for chunk in target_chunks:
        cid = str(chunk["chunk_id"])
        if os.path.exists(os.path.join(OUTPUT_DIR, f"{cid}.txt")):
            skipped += 1
        else:
            pending.append(chunk)

    if skipped:
        print(f"Skipping {skipped} already-completed chunks.")
    print(f"Processing {len(pending)} chunks.\n")

    def task(chunk):
        cid = str(chunk["chunk_id"])
        idx = chunk_index.get(cid, 0)
        prev_chunk    = full_sorted[idx - 1] if idx > 0 else None
        prev_chunk_id = str(prev_chunk["chunk_id"]) if prev_chunk else None
        return arbitrate_chunk(chunk, prev_chunk_id)

    times  = []
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, c): c["chunk_id"] for c in pending}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Differ"):
            chunk_id, elapsed, error = future.result()
            if error:
                errors.append((chunk_id, error))
                tqdm.write(f"  [{chunk_id}] ERROR: {error}")
            elif elapsed is not None:
                tqdm.write(f"  [{chunk_id}] {elapsed:.1f}s")
                times.append(elapsed)

    print(f"\n=== SUMMARY ===")
    if times:
        print(f"Completed: {len(times)}/{len(pending)} chunks")
        print(f"Total: {sum(times):.1f}s | Avg: {sum(times)/len(times):.1f}s")
    print(f"Pre-existing: {skipped}")
    if errors:
        print(f"Errors: {len(errors)}")
        for cid, err in errors:
            print(f"  [{cid}] {err}")


if __name__ == "__main__":
    start   = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end     = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    run(start, end, workers)
