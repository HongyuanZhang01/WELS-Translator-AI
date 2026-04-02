#!/usr/bin/env python3
"""
build_walther_glossaries.py

Translates theological_vocabulary_deduped.json into 9 target languages,
writing one incrementally-updated JSON glossary per language to:
    ~/ge_rig/output/glossaries/glossary_<code>.json

Each output entry preserves:
    term, english, lutheran_nuance, translation_note  (source fields)
    <lang>                                             (translated term)
    alternatives                                       (0-2 alternatives)
    note                                               (translator commentary)

Concurrency model
-----------------
  LANG_WORKERS  : languages processed in parallel
  TERM_WORKERS  : terms queried in parallel within each language
  Max in-flight : LANG_WORKERS × TERM_WORKERS

Recommended starting point: LANG_WORKERS=3, TERM_WORKERS=16 → 48 concurrent
requests. These are small, fast calls (max_tokens=300) so you can push
TERM_WORKERS significantly higher than the annotation pipeline.

Resume safety
-------------
Each language has its own output file. Re-running skips any term already
present in the output JSON, so partial runs resume cleanly.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse

from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000/v1"
API_KEY  = "token-not-needed"
MODEL    = "Qwen/Qwen3.5-35B-A3B-FP8"

INPUT_JSON  = Path.home() / "ge_rig" / "output" / "theological_vocabulary_deduped.json"
OUTPUT_DIR  = Path.home() / "ge_rig" / "output" / "glossaries"

MAX_TOKENS   = 350    # slightly larger to accommodate nuance commentary
LANG_WORKERS = 1      # languages processed in parallel
TERM_WORKERS = 48     # terms queried in parallel within each language
MAX_RETRIES  = 3
RETRY_DELAY  = 2.0

# ── System prompt factory ──────────────────────────────────────────────────────

def _make_system_prompt(language: str, field: str, extra_guidance: str = "") -> str:
    base = (
        f"You are a specialist translator of 19th-century Lutheran theological German "
        f"into {language}, working on C. F. W. Walther's *The Proper Distinction Between "
        f"Law and Gospel*.\n\n"
        f"You will receive a German theological term together with:\n"
        f"  - an English gloss\n"
        f"  - a Lutheran nuance note explaining confessional precision\n"
        f"  - a translation note with specific warnings for translators\n\n"
        f"Use all three to inform your translation. Respond ONLY with a JSON object — "
        f"no markdown fences, no extra keys. The object must contain exactly these keys:\n"
        f'  "term": the original German term, copied through verbatim,\n'
        f'  "english": the English gloss, copied through verbatim,\n'
        f'  "lutheran_nuance": the Lutheran nuance, copied through verbatim,\n'
        f'  "translation_note": the translation note, copied through verbatim,\n'
        f'  "{field}": the best {language} rendering of the term — a single word or '
        f"short nominal phrase only, not a definition or explanatory clause,\n"
    )
    hierarchy = (
        f"Use Lutheran terminology where available; fall back to standard Reformed / "
        f"Protestant vocabulary (preferring established Bible translation usage) when "
        f"necessary; use Catholic or other Christian terminology only as a final resort, "
        f"and flag it explicitly in the note when you do so.\n\n"
    )
    fields = (
        f'  "alternatives": a list of 0–2 alternative {language} renderings (may be empty),\n'
        f'  "note": one sentence of translator\'s commentary in English covering register, '
        f"confessional nuance, any deviation from the translation_note's guidance, or "
        f"script/orthography notes relevant to a {language}-speaking Lutheran reader. "
        f"Empty string if nothing material to add."
    )
    return base + hierarchy + (extra_guidance + "\n\n" if extra_guidance else "") + fields


# ── Language table ─────────────────────────────────────────────────────────────

LANGUAGES = [
    (
            "hu", "Hungarian", "hungarian",
            _make_system_prompt(
            "Hungarian", "hungarian",
            "Prefer terminology from the Hungarian Lutheran (Magyarországi Evangélikus Egyház) "
            "tradition. The Károli Bible serves as the Protestant baseline for scriptural "
            "vocabulary. "
            "Key fixed renderings to observe: Gesetz = törvény, Evangelium = evangélium, "
            "Gnade = kegyelem, Glaube = hit, Rechtfertigung = megigazulás, "
            "Heiligung = megszentelődés, Buße = bűnbánat, Gewissen = lelkiismeret, "
            "Seligkeit = üdvösség (not boldogság), Selig = üdvözült (not boldog), "
            "Sakrament = szentség, Taufe = keresztség, Verdammnis = kárhozat. "
            "Flag any term where Catholic Hungarian usage (e.g. from the Neovulgata tradition) "
            "differs significantly from Lutheran usage. "
            "Avoid Reformed Hungarian coinages where a distinct Lutheran term exists."
        ),
    ),
    (
        "es", "Spanish", "spanish",
        _make_system_prompt(
            "Spanish", "spanish",
            "Prefer Reina-Valera vocabulary as the Reformed/Protestant baseline. "
            "Use Latin American register (ustedes, not vosotros) unless the term is "
            "specifically Peninsular. Flag when Catholic terminology is used as a fallback."
        ),
    ),
    (
        "pt", "Portuguese", "portuguese",
        _make_system_prompt(
            "Portuguese", "portuguese",
            "Prefer Brazilian Portuguese register and Protestant Bible translation "
            "vocabulary (e.g. João Ferreira de Almeida tradition) as the Reformed baseline. "
            "Flag when Catholic terminology is used as a fallback."
        ),
    ),
    (
        "ru", "Russian", "russian",
        _make_system_prompt(
            "Russian", "russian",
            "Prefer terminology used in Russian Lutheran and Protestant communities. "
            "The Synodal Bible translation may serve as the Reformed/Protestant baseline. "
            "Russian Orthodox terminology may be used as a last resort; flag it explicitly "
            "in the note. Output Russian in Cyrillic script."
        ),
    ),
    (
        "uk", "Ukrainian", "ukrainian",
        _make_system_prompt(
            "Ukrainian", "ukrainian",
            "Prefer terminology used in Ukrainian Lutheran and Protestant communities. "
            "The Ogienko or Khomenko Bible translations may serve as the baseline. "
            "Ukrainian Orthodox terminology may be used as a last resort; flag it in the note. "
            "Output Ukrainian in Cyrillic script."
        ),
    ),
    (
        "sw", "Swahili", "swahili",
        _make_system_prompt(
            "Swahili", "swahili",
            "Prefer terminology from East African Lutheran communities (Tanzania, Kenya). "
            "The Union Version Swahili Bible (Biblia Habari Njema or the older Swahili Union) "
            "may serve as the Protestant baseline. Avoid Roman Catholic Swahili equivalents "
            "unless no Protestant term exists; flag when used."
        ),
    ),
    (
        "fr", "French", "french",
        _make_system_prompt(
            "French", "french",
            "Prefer terminology from French Protestant/Reformed tradition "
            "(e.g. Louis Segond Bible) as the baseline. "
            "Flag when Catholic French terminology is used as a fallback."
        ),
    ),
    (
        "ne", "Nepali", "nepali",
        _make_system_prompt(
            "Nepali", "nepali",
            "Prefer terminology used in Nepali Protestant and Lutheran communities. "
            "The Nepali Bible (Parbatiya Bible / modern translations) may serve as baseline. "
            "Output Nepali in Devanagari script."
        ),
    ),
    (
        "zh", "Mandarin Chinese", "mandarin",
        _make_system_prompt(
            "Mandarin Chinese", "mandarin",
            "Prefer Simplified Chinese characters and terminology used in mainland Chinese "
            "Protestant communities. The Chinese Union Version (和合本, Héhé Běn) serves as "
            "the Protestant baseline. Traditional characters may be listed as an alternative "
            "where they differ significantly. Flag when Catholic Chinese terminology is used."
        ),
    ),
    (
        "id", "Indonesian", "indonesian",
        _make_system_prompt(
            "Indonesian", "indonesian",
            "Prefer terminology from Indonesian Protestant/Lutheran communities. "
            "The Indonesian Terjemahan Baru (TB) Bible serves as the Protestant baseline. "
            "Flag when Catholic Indonesian terminology is used as a fallback."
        ),
    ),
]

# ── vLLM client ────────────────────────────────────────────────────────────────

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_user_prompt(entry: dict, field: str) -> str:
    return (
        f"German term: {entry['term']}\n"
        f"English gloss: {entry.get('english', '')}\n"
        f"Lutheran nuance: {entry.get('lutheran_nuance', '')}\n"
        f"Translation note: {entry.get('translation_note', '')}\n\n"
        f"Return the JSON object described in the system prompt. "
        f"The \"{field}\" field must be a short term only, not a definition."
    )


def load_existing(path: Path) -> dict:
    """Load existing glossary; return dict keyed by term."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return {e["term"]: e for e in data if "term" in e}
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


def save_all(path: Path, glossary: dict, lock: threading.Lock) -> None:
    """Atomically write glossary to disk."""
    tmp = str(path) + ".tmp"
    with lock:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(list(glossary.values()), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


def query_term(entry: dict, field: str, system_prompt: str, lang_label: str) -> dict:
    """Query the model for a single term; retry on failure."""
    term = entry["term"]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": make_user_prompt(entry, field)},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.15,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            )
            raw = response.choices[0].message.content.strip()

            # Strip accidental markdown fences
            raw = raw.replace("```json", "").replace("```", "").strip()

            result = json.loads(raw)

            # Guarantee source fields are preserved correctly
            result["term"]             = term
            result["english"]          = entry.get("english", "")
            result["lutheran_nuance"]  = entry.get("lutheran_nuance", "")
            result["translation_note"] = entry.get("translation_note", "")

            return result

        except json.JSONDecodeError as exc:
            print(f"  [{lang_label}] JSON error on '{term}' (attempt {attempt}): {exc}")
        except Exception as exc:
            print(f"  [{lang_label}] Request error on '{term}' (attempt {attempt}): {exc}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    # Fallback entry on total failure
    return {
        "term":             term,
        "english":          entry.get("english", ""),
        "lutheran_nuance":  entry.get("lutheran_nuance", ""),
        "translation_note": entry.get("translation_note", ""),
        field:              "",
        "alternatives":     [],
        "note":             f"ERROR: failed after {MAX_RETRIES} attempts",
    }


def process_language(
    entries: list,
    code: str,
    language: str,
    field: str,
    system_prompt: str,
) -> None:
    out_path   = OUTPUT_DIR / f"glossary_{code}.json"
    write_lock = threading.Lock()
    glossary   = load_existing(out_path)
    done_terms = set(glossary.keys())

    pending = [e for e in entries if e["term"] not in done_terms]

    print(f"[{language}] {len(glossary)} done, {len(pending)} remaining → {out_path.name}")

    if not pending:
        print(f"[{language}] Already complete, skipping.")
        return

    def _process_entry(entry: dict) -> None:
        result = query_term(entry, field, system_prompt, language)
        glossary[entry["term"]] = result
        save_all(out_path, glossary, write_lock)
        translated = result.get(field, "?")
        print(f"  [{language}] ✓ {entry['term']}  →  {translated}")

    with ThreadPoolExecutor(max_workers=TERM_WORKERS) as pool:
        futures = {pool.submit(_process_entry, entry): entry for entry in pending}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                entry = futures[fut]
                print(f"  [{language}] ERROR on '{entry['term']}': {exc}")

    print(f"[{language}] Done. {len(glossary)} entries written.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Walther multilingual glossaries")
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Only process this language code (e.g. --lang hu). Omit to run all."
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        entries = json.load(f)
    entries = [e for e in entries if isinstance(e, dict) and e.get("term")]

    # Filter languages if --lang specified
    languages = LANGUAGES
    if args.lang:
        languages = [l for l in LANGUAGES if l[0] == args.lang]
        if not languages:
            valid = [l[0] for l in LANGUAGES]
            print(f"Unknown language code '{args.lang}'. Valid codes: {valid}")
            return

    active_lang_workers = min(LANG_WORKERS, len(languages))

    print(f"Loaded {len(entries)} terms from {INPUT_JSON}")
    print(f"Languages: {len(languages)}  |  "
          f"Lang workers: {active_lang_workers}  |  "
          f"Term workers: {TERM_WORKERS}  |  "
          f"Max concurrent requests: {active_lang_workers * TERM_WORKERS}\n")

    with ThreadPoolExecutor(max_workers=active_lang_workers) as pool:
        futures = {
            pool.submit(process_language, entries, code, language, field, prompt): language
            for code, language, field, prompt in languages
        }
        for fut in as_completed(futures):
            lang = futures[fut]
            exc  = fut.exception()
            if exc:
                print(f"[{lang}] FATAL ERROR: {exc}")

    print("\nAll languages complete.")


if __name__ == "__main__":
    main()