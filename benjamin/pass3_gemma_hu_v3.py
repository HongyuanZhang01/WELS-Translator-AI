"""
pass3_gemma_hu.py  (v2 — lean few-shot prompt)
Pass 3 — Editorial refinement and production finalisation of the Pass 2
Hungarian translation of Walther's *The Proper Distinction Between Law and Gospel*.

Changes from v1:
  - System prompt stripped to essentials + 6 few-shot examples
  - Glossary injected directly from Pass 2 debug output (the term list
    appended after the === separator in each pass2 .txt file)
  - No hard-coded priority term list; glossary is whatever Pass 2 selected
  - Model configurable via MODEL constant — swap for Qwen/Mistral testing

Input:
  - chunks.json                  : JSON array of {chunk_id, lecture, thesis, text}
  - hu_qwen/pass2/<id>.txt       : Pass 2 output (may contain debug glossary block)
  - hu_gemma/pass3/<prev_id>.txt : previous chunk finalised Hungarian (for tail)

Output:
  - hu_gemma/pass3/<id>.txt      : production-ready Hungarian

Usage:
  python pass3_gemma_hu.py [start] [end] [max_workers]
  python pass3_gemma_hu.py 1 50 3
"""

import sys
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR    = os.path.expanduser("~/ge_rig")
CHUNKS_FILE = os.path.join(BASE_DIR, "output", "chunks.json")
PASS2_DIR   = os.path.join(BASE_DIR, "output", "hu_qwen", "gemma", "pass2")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output", "hu_gemma", "pass3b")

MODEL  = "RedHatAI/gemma-3-27b-it-FP8-dynamic"
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-not-needed")

# Characters of previous finalised Hungarian chunk to include for continuity
PREV_HU_TAIL_CHARS = 600

# Separator used in Pass 2 debug output to mark the glossary block
PASS2_GLOSSARY_SEP = "DEBUG — GLOSSARY TERMS INJECTED FOR THIS CHUNK"

# =============================================================================
# SYSTEM PROMPT — lean, few-shot anchored
# =============================================================================

SYSTEM_PROMPT = """\
You are a Hungarian Lutheran theological editor.

You are working on a draft of C. F. W. Walther's *The Proper Distinction Between Law and Gospel translated from 19th-century Lutheran German. A machine translation pipeline has produced a semantically accurate Hungarian draft. Your task is to make it completely publication-ready.

You will receive only the PASS 2 DRAFT — the Hungarian text to finalise. You must make it into completely fluent, native-sounding Hungarian without changing the meaning. You must take special care to preserve the theological sense and key theological words.

Consider the following as you work:
1. Sentences should follow normal, authentic Hungarian word-order. They should not simply reproduce the German flow.
2. In places where duplicate terms appear, we may use an appropriate synonym instead.
3. Theological terms should match the standard Lutheran usage. If you aren't sure about a technical term, do not change it.
4. Biblical quotes and references should stick to the standard Károli form.
5. No German terms should leak through untranslated.
6. Latin, Greek, and Hebrew technical terms should be left in their original language with a Hungarian translation in parentheses.
7. All Hungarian grammar should be perfectly standard.
8. Broken, hallucinated, or made-up words should be fixed.
9. If a word makes no sense in context, try to reconstruct the error and replace with the word you think is accurate.
10. The tone should match a formal pastoral sermon or theological lecture.
11. The formal Ön should be used throughout.
12. Do not smooth out rhetorical force - make sure the emotion comes through cleanly.


=== FEW-SHOT EXAMPLES ===

--- EXAMPLE 1: Morphological error and wrong glossary rendering ---

DRAFT:
Az isteni törvény büntetéstöl és átkozottságától csak a pokolban fog az ember \
megszabadulni; mert a törvénynek teljesülnie kell.

CORRECTED:
Az isteni törvény büntetésétől és átkától csak a pokolban fog az ember \
megszabadulni; mert a törvénynek teljesülnie kell.

WHAT CHANGED:
— "büntetéstöl" → "büntetésétől" (wrong vowel harmony in case suffix)
— "átkozottságától" → "átkától" (átok is the correct Lutheran Hungarian \
  rendering for Fluch; átkozottság means 'the state of being cursed' which \
  shifts agency away from God's pronouncement)

--- EXAMPLE 2: Infinitive standing as finite imperative ---

DRAFT:
De azért kétségbeesni ne! Valaki megszerezte neked az üdvösséget.

CORRECTED:
De azért ne essetek kétségbe! Valaki megszerezte neked az üdvösséget.

WHAT CHANGED:
— "kétségbeesni ne" → "ne essetek kétségbe" (Hungarian does not permit an \
  infinitive as a finite negative imperative; the correct plural imperative \
  is "ne essetek kétségbe")

--- EXAMPLE 3: German calque ---

DRAFT:
Amíg az ember még testi kondícióban van a bűneiben, addig csak az átkozó \
törvényt kell hirdetni.

CORRECTED:
Amíg az ember még jól van a bűneiben, addig csak az átkozó törvényt kell \
hirdetni.

WHAT CHANGED:
— "testi kondícióban van" → "jól van" (direct calque of German "wohl ist \
  in seinen Sünden"; the established Hungarian idiom is "jól van a bűneiben")

--- EXAMPLE 4: Garbled phrase containing a non-word ---

DRAFT:
Hasonló az emberhez, aki magához toll és tele ivott, és amikor a legdrágább \
ételt teszik elé, csak annyit mond: „Hm!"

CORRECTED:
Hasonló az emberhez, aki tele evett és ivott magát, és amikor a legdrágább \
ételt teszik elé, csak annyit mond: „Hm!"

WHAT CHANGED:
— "magához toll és tele ivott" → "tele evett és ivott magát" ("toll" is a \
  non-word in this context; the phrase garbles the German "toll und voll" — \
  the correct Hungarian rendering is "tele evett és ivott magát")

--- EXAMPLE 5: Károli Bible alignment ---

DRAFT:
Péter törvényprédikációja után így áll: „Ekkor szúrta át a szívüket." \
De aztán megkérdezték: „Ti férfiak, mit cselekedjünk?"

CORRECTED:
Péter törvényprédikációja után így áll: „Szívökbe hatott." \
De aztán megkérdezték: „Ti férfiak, mit cselekedjünk?"

WHAT CHANGED:
— "Ekkor szúrta át a szívüket" → "Szívökbe hatott" (Károli Bible Acts 2:37; \
  direct scripture quotations must match established Károli phrasing)

"""

# =============================================================================
# PASS 2 OUTPUT PARSING
# Splits the Pass 2 .txt file into the translation body and the debug glossary
# block (if present). The debug block starts after the PASS2_GLOSSARY_SEP line.
# =============================================================================

def parse_pass2_output(raw: str) -> tuple[str, str]:
    """
    Returns (translation_body, glossary_block).
    If no debug separator is present, glossary_block is empty string.
    """
    sep_marker = "=" * 60
    lines = raw.split("\n")
    sep_indices = [i for i, l in enumerate(lines) if l.strip() == sep_marker]

    if len(sep_indices) >= 1:
        # First separator marks start of debug block
        body_lines = lines[:sep_indices[0]]
        gloss_lines = []
        # Find the line after the header (skip the two separator lines and header)
        in_gloss = False
        for line in lines[sep_indices[0]:]:
            if PASS2_GLOSSARY_SEP in line:
                in_gloss = True
                continue
            if in_gloss and line.strip() == sep_marker:
                continue
            if in_gloss:
                gloss_lines.append(line)
        return "\n".join(body_lines).strip(), "\n".join(gloss_lines).strip()

    return raw.strip(), ""


# =============================================================================
# MODEL CALL
# =============================================================================

def _call(messages: list, timeout: int = 600) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=timeout,
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# PASS 3 LOGIC
# =============================================================================

def refine_chunk(chunk: dict, prev_chunk_id: str | None) -> tuple:

    chunk_id   = str(chunk["chunk_id"])
    pass2_path = os.path.join(PASS2_DIR, f"{chunk_id}.txt")
    if not os.path.exists(pass2_path):
        return chunk_id, None, f"Pass 2 output missing for {chunk_id}"

    with open(pass2_path, "r", encoding="utf-8") as f:
        raw = f.read()

    out_path = os.path.join(OUTPUT_DIR, "pass3", f"{chunk_id}.txt")
    if os.path.exists(out_path):
        return chunk_id, None, None

    pass2_draft, glossary_block = parse_pass2_output(raw)

    # Assemble user content
    user_content = ""

    user_content += f"PASS 2 DRAFT:\n{pass2_draft}\n\n"
    user_content += "Produce the finalised Hungarian text now."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    try:
        t0      = time.time()
        result  = _call(messages, timeout=600)
        elapsed = time.time() - t0
        os.makedirs(os.path.join(OUTPUT_DIR, "pass3"), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)
        return chunk_id, elapsed, None
    except Exception as e:
        return chunk_id, None, f"Pass 3 error: {e}"


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(start_idx: int, end_idx: int, max_workers: int):
    os.makedirs(os.path.join(OUTPUT_DIR, "pass3"), exist_ok=True)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    all_chunks.sort(key=lambda c: int(str(c["chunk_id"])))
    target_chunks = all_chunks[start_idx - 1 : end_idx]

    full_sorted = sorted(all_chunks, key=lambda c: int(str(c["chunk_id"])))
    chunk_index = {str(c["chunk_id"]): i for i, c in enumerate(full_sorted)}

    print(f"Pass 3 refinement | Hungarian | chunks {start_idx}–{end_idx} "
          f"({len(target_chunks)} chunks) | workers: {max_workers}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUTPUT_DIR}/pass3\n")

    pending, skipped = [], 0
    for chunk in target_chunks:
        cid = str(chunk["chunk_id"])
        if os.path.exists(os.path.join(OUTPUT_DIR, "pass3", f"{cid}.txt")):
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
        return refine_chunk(chunk, prev_chunk_id)

    times  = []
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, c): c["chunk_id"] for c in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Pass 3"):
            chunk_id, elapsed, error = future.result()
            if error:
                errors.append((chunk_id, error))
                tqdm.write(f"  [{chunk_id}] ERROR: {error}")
            elif elapsed is not None:
                tqdm.write(f"  [{chunk_id}] {elapsed:.1f}s")
                times.append(elapsed)

    print(f"\n=== SUMMARY ===")
    if times:
        print(f"Completed: {len(times)}/{len(pending)} chunks in this run")
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
