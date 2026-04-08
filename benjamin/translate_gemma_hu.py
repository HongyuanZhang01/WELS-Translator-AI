"""
translate_gemma_hu.py
Independent Hungarian translation using TranslateGemma for differ architecture.

This script produces a COMPLETELY INDEPENDENT Hungarian translation from the
German source — no MadLAD draft, no Qwen output in context. The output is
used as a second candidate for comparison against the Qwen Pass 2 output.

Sentences where both models agree → almost certainly correct.
Sentences where they diverge → flagged for deterministic resolution or review.

TranslateGemma prompt structure:
  <<<source>>>de<<<target>>>hu<<<text>>>
  [INSTRUCTIONS]...[/INSTRUCTIONS]
  [GLOSSARY]...[/GLOSSARY]
  Text: <german source>

Notes:
  - TranslateGemma follows instructions less flexibly than instruct models.
    Keep prompts lean — the format does the work.
  - Glossary injection at full length is problematic. We inject only the
    highest-priority terms matched to the chunk via Levenshtein lookup,
    with a tighter cap than Pass 2.
  - No thinking mode, no system prompt.

Input:
  - chunks.json                    : JSON array of {chunk_id, lecture, thesis, text}
  - hu_gemma/translate/<prev>.txt  : previous chunk's output (tail only)
  - theological_vocabulary_deduped.json : glossary

Output:
  - hu_gemma/translate/<id>.txt    : independent Hungarian translation

Usage:
  python translate_gemma_hu.py [start] [end] [max_workers]
  python translate_gemma_hu.py 1 50 4
"""

import sys
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from rapidfuzz.distance import Levenshtein
from openai import OpenAI
from tqdm import tqdm
import spacy

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR      = os.path.expanduser("~/ge_rig")
CHUNKS_FILE   = os.path.join(BASE_DIR, "output", "chunks.json")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output", "hu_gemma")
GLOSSARY_PATH = os.path.join(BASE_DIR, "output", "theological_vocabulary_deduped.json")

MODEL  = "Infomaniak-AI/vllm-translategemma-27b-it"   # update to your served model name
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-not-needed")

# Characters of previous chunk output to include for style continuity
PREV_HU_TAIL_CHARS = 400

# Glossary lookup — tighter cap than Pass 2 given TranslateGemma constraints
MAX_GLOSSARY_TERMS   = 40
MAX_PER_SOURCE_TOKEN = 2

# =============================================================================
# SPACY + GLOSSARY
# =============================================================================

_nlp      = None
_nlp_lock = threading.Lock()

def _get_nlp():
    global _nlp
    with _nlp_lock:
        if _nlp is None:
            _nlp = spacy.load("de_core_news_sm")
    return _nlp


_GLOSSARY_CACHE = None
_GLOSSARY_LOCK  = threading.Lock()

def _load_glossary() -> list:
    global _GLOSSARY_CACHE
    with _GLOSSARY_LOCK:
        if _GLOSSARY_CACHE is None:
            if os.path.exists(GLOSSARY_PATH):
                with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
                    _GLOSSARY_CACHE = json.load(f)
            else:
                _GLOSSARY_CACHE = []
    return _GLOSSARY_CACHE


_FUNCTION_POS     = {"ADP", "DET", "PRON", "CCONJ", "SCONJ", "PART", "PUNCT", "SPACE", "AUX"}
_GLOSS_INDEX_CACHE = None
_GLOSS_INDEX_LOCK  = threading.Lock()

def _build_glossary_index(glossary: list, lang_field: str = "hungarian"):
    single_index   = {}
    phrase_entries = []
    nlp = _get_nlp()
    for entry in glossary:
        if not isinstance(entry, dict):
            continue
        term = entry.get("term", "").strip()
        hu   = entry.get(lang_field, "").strip()
        if not term or not hu:
            continue
        words = term.split()
        if len(words) == 1:
            single_index[term.lower()] = entry
        else:
            doc    = nlp(term)
            lemmas = frozenset(
                t.lemma_.lower() for t in doc
                if t.pos_ not in _FUNCTION_POS and len(t.lemma_) > 1
            )
            if lemmas:
                phrase_entries.append((lemmas, len(words) + 3, entry))
    return single_index, phrase_entries


def _get_glossary_index(lang_field: str = "hungarian"):
    global _GLOSS_INDEX_CACHE
    with _GLOSS_INDEX_LOCK:
        if _GLOSS_INDEX_CACHE is None:
            _GLOSS_INDEX_CACHE = _build_glossary_index(_load_glossary(), lang_field)
    return _GLOSS_INDEX_CACHE


def get_relevant_glossary(german_text: str, lang_field: str = "hungarian") -> str:
    single_index, phrase_entries = _get_glossary_index(lang_field)
    nlp = _get_nlp()

    doc = nlp(german_text)
    source_tokens      = [t for t in doc if not t.is_punct and not t.is_space]
    source_lemmas_list = [t.lemma_.lower() for t in source_tokens]
    source_lemmas_set  = set(source_lemmas_list)

    results        = []
    seen_headwords = set()

    # Phase 1: phrase matching
    for content_lemmas, window, entry in phrase_entries:
        if len(results) >= MAX_GLOSSARY_TERMS:
            break
        term = entry.get("term", "").lower()
        if term in seen_headwords:
            continue
        if not content_lemmas.issubset(source_lemmas_set):
            continue
        for i in range(len(source_lemmas_list)):
            if content_lemmas.issubset(set(source_lemmas_list[i: i + window])):
                seen_headwords.add(term)
                results.append(entry)
                break

    # Phase 2: single-token Levenshtein
    seen_source = set()
    headwords   = list(single_index.keys())

    for lemma in source_lemmas_list:
        if lemma in seen_source or len(results) >= MAX_GLOSSARY_TERMS:
            break
        seen_source.add(lemma)
        n     = len(lemma)
        max_d = 1 if n <= 4 else (2 if n <= 7 else 3)

        candidates = []
        for hw in headwords:
            if hw in seen_headwords:
                continue
            dist = Levenshtein.distance(lemma, hw, score_cutoff=max_d)
            if dist <= max_d:
                candidates.append((dist, hw))

        candidates.sort(key=lambda x: x[0])
        for _, hw in candidates[:MAX_PER_SOURCE_TOKEN]:
            if hw not in seen_headwords and len(results) < MAX_GLOSSARY_TERMS:
                seen_headwords.add(hw)
                results.append(single_index[hw])

    if not results:
        return ""

    lines = []
    for entry in results:
        term = entry.get("term", "")
        hu   = entry.get(lang_field, "")
        if term and hu:
            lines.append(f'"{term}" -> "{hu}"')
    return "\n".join(lines)


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt(
    german_text: str,
    glossary_block: str,
    prev_hu_tail: str,
) -> str:
    instructions = [
        "- Style: 19th-century Lutheran theological lecture",
        "- Target register: Károli Bible Hungarian",
        "- Preserve Latin, Greek, Hebrew terms unchanged",
        "- Preserve all citation references unchanged",
    ]

    if prev_hu_tail:
        # Brief tail anchor — just the last sentence or two
        tail = prev_hu_tail[-200:].strip()
        instructions.append(f"- Continue naturally from: ...{tail}")

    instructions_str = "\n".join(instructions)

    prompt = (
        f"<<<source>>>de<<<target>>>hu<<<text>>>"
        f"[INSTRUCTIONS]\n{instructions_str}\n[/INSTRUCTIONS]\n\n"
    )

    if glossary_block:
        prompt += f"[GLOSSARY]\n{glossary_block}\n[/GLOSSARY]\n\n"

    prompt += f"Text: {german_text}"
    return prompt


# =============================================================================
# MODEL CALL
# =============================================================================

def _call(prompt: str, timeout: int = 600) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        timeout=timeout,
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# TRANSLATION LOGIC
# =============================================================================

def translate_chunk(chunk: dict, prev_chunk_id: str | None) -> tuple:
    chunk_id   = str(chunk["chunk_id"])
    chunk_text = chunk["text"].strip()

    out_path = os.path.join(OUTPUT_DIR, "translate", f"{chunk_id}.txt")
    if os.path.exists(out_path):
        return chunk_id, None, None

    # Previous chunk tail for style continuity
    prev_hu_tail = ""
    if prev_chunk_id is not None:
        prev_path = os.path.join(OUTPUT_DIR, "translate", f"{prev_chunk_id}.txt")
        if os.path.exists(prev_path):
            with open(prev_path, "r", encoding="utf-8") as f:
                prev_hu = f.read().strip()
            if prev_hu:
                prev_hu_tail = prev_hu[-PREV_HU_TAIL_CHARS:]

    glossary_block = get_relevant_glossary(chunk_text)
    prompt         = build_prompt(chunk_text, glossary_block, prev_hu_tail)

    try:
        t0      = time.time()
        result  = _call(prompt, timeout=600)
        elapsed = time.time() - t0

        os.makedirs(os.path.join(OUTPUT_DIR, "translate"), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)

        return chunk_id, elapsed, None
    except Exception as e:
        return chunk_id, None, f"TranslateGemma error: {e}"


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(start_idx: int, end_idx: int, max_workers: int):
    os.makedirs(os.path.join(OUTPUT_DIR, "translate"), exist_ok=True)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    all_chunks.sort(key=lambda c: int(str(c["chunk_id"])))
    target_chunks = all_chunks[start_idx - 1 : end_idx]

    full_sorted = sorted(all_chunks, key=lambda c: int(str(c["chunk_id"])))
    chunk_index = {str(c["chunk_id"]): i for i, c in enumerate(full_sorted)}

    print(f"TranslateGemma | Hungarian | chunks {start_idx}–{end_idx} "
          f"({len(target_chunks)} chunks) | workers: {max_workers}")
    print(f"Model:  {MODEL}")
    print(f"Output: {OUTPUT_DIR}/translate\n")

    pending, skipped = [], 0
    for chunk in target_chunks:
        cid = str(chunk["chunk_id"])
        if os.path.exists(os.path.join(OUTPUT_DIR, "translate", f"{cid}.txt")):
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
        return translate_chunk(chunk, prev_chunk_id)

    times  = []
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, c): c["chunk_id"] for c in pending}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="TranslateGemma"):
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
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    run(start, end, workers)
