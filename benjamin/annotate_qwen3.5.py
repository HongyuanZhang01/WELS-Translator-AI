"""
annotate_qwen3_5_hu.py
Four-pass annotation pipeline for Luther's Large Catechism chunks.

Passes (sequential per chunk, chunks processed concurrently):
  1. Overview   – summary, tone, argument structure
  2. Theology   – theologically significant vocabulary with Lutheran nuance
  3. Vocabulary – other rare / challenging / archaic vocabulary
  4. Rhetoric   – metaphors, euphemisms, word-pictures (16th-c. context)

Usage:
  python annotate_qwen3_5_hu.py [start] [end] [--batch-size N]

  start / end   : 1-based chunk indices (default: 1 / all)
  --batch-size  : concurrent chunks (default: 4)
"""

import sys
import os
import json
import json_repair
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
with open('/home/benjamin/translation/hungarian_vocab.json', 'r', encoding='utf-8') as f:
    GLOSSARY = json.load(f)

CHUNKS_FILE = os.path.expanduser("~/ge_rig/output/chunks.json")
OUTPUT_DIR = os.path.expanduser("~/ge_rig/output/annotated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-not-needed")

MODEL            = "Qwen/Qwen3.5-35B-A3B-FP8"
TEMPERATURE      = 0.3
THINKING_BUDGET  = 4048   # tokens; raise to 1024 on Pass 4 if rhetorical quality suffers
TIMEOUT          = 800   # seconds per call
PASS_THINKING = {
    1: False,   # argument structure benefits from reasoning
    2: False,   # theological nuance definitely benefits
    3: False,  # straightforward lexical extraction
    4: False,   # rhetorical analysis benefits
}

# ---------------------------------------------------------------------------
# System prompts – one per pass, each tight and focused
# ---------------------------------------------------------------------------

SYSTEM_PASS1 = """\
You are a scholarly annotator working on C. F. W. Walther's *The Proper Distinction Between Law and Gospel* (19th century German).
Your task: produce a structured overview of the passage.

Return ONLY valid JSON with exactly these keys:
{
  "summary": "<3-4 sentence English summary of the passage's content and purpose>",
  "tone": "<3-6 descriptive words, e.g. 'pastoral, urgent, polemical'>",
  "argument_structure": [
    {"step": 1, "description": "<what Luther is doing rhetorically/argumentatively>"},
    ...
  ]
}

No prose outside the JSON object. No markdown fences."""

SYSTEM_PASS2 = """\
You are a Lutheran theological lexicographer working on C. F. W. Walther's *The Proper Distinction Between Law and Gospel* (19th century German).
Your task: extract ALL theologically significant vocabulary from the passage.

"Theologically significant" means: terms that carry confessional weight in a Lutheran context, \
including Law/Gospel distinctions, sacramental terms, soteriological vocabulary, \
ecclesiological terms, polemical labels (e.g. Schwärmer, Papisten), and scriptural allusions.

Return ONLY valid JSON — an object with a single key "theological_vocabulary" containing an array:
{
  "theological_vocabulary": [
    {
      "term": "<German word or phrase>",
      "english": "<English gloss, 1-10 words>",
      "lutheran_nuance": "<confessional precision: what this term means specifically in Lutheran theology, 10-25 words>",
      "translation_note": "<any warning for translators, i.e., meanings to preserve or avoid, 10-15 words>"
    },
    ...
  ]
}

Be exhaustive. Do not skip terms because they seem obvious.
No prose outside the JSON object. No markdown fences."""

SYSTEM_PASS3 = """\
You are a philologist working on C. F. W. Walther's *The Proper Distinction Between Law and Gospel* (19th century German).
Your task: extract ALL vocabulary that is rare, archaic, legally or scholastically technical, \
or otherwise likely to challenge a modern translator — EXCLUDING terms already covered \
by theological vocabulary (Law/Gospel, sacraments, soteriology, ecclesiology).

Return ONLY valid JSON — an object with a single key "other_vocabulary" containing an array:
{
  "other_vocabulary": [
    {
      "term": "<German word or phrase>",
      "english": "<English gloss, 1-10 words>",
      "nuance": "<why this is difficult; register, archaic usage, legal/scholastic origin, 10-20 words>",
      "translation_note": "<practical note for the translator>"
    },
    ...
  ]
}

Be exhaustive, but include no more than 20 terms. No prose outside the JSON object. No markdown fences."""

SYSTEM_PASS4 = """\
You are a literary analyst specializing in 16th-century German prose rhetoric.
Your task: identify and explain ALL metaphors, similes, euphemisms, word-pictures, \
analogies, and concrete images in the passage.

Keep the 16th-century context firmly in mind: images drawn from guild life, legal proceedings, \
household economy, agriculture, warfare, and ecclesiastical ceremony are intentional and carry \
rhetorical weight. Do not flatten or modernize them in your explanations.

Return ONLY valid JSON — an object with a single key "rhetorical_devices" containing an array:
{
  "rhetorical_devices": [
    {
      "german_phrase": "<the original German phrase>",
      "device_type": "<metaphor | simile | euphemism | word-picture | analogy | irony | other>",
      "english_explanation": "<what the image is and what it communicates, 10-20 words>",
      "sixteenth_century_context": "<what a contemporary reader would have understood that a modern reader might miss, 10-25 words>",
      "translation_imperative": "<what must be preserved or avoided in translation>"
    },
    ...
  ]
}

No prose outside the JSON object. No markdown fences."""

# ---------------------------------------------------------------------------
# Few-shot examples (serialized correctly as JSON strings)
# ---------------------------------------------------------------------------

FEW_SHOT_USER = (
    "Das ist freilich wahr: Die lutherische Kirche hebt die Sacramente überaus hoch, "
    "weil sie dieselben nicht für bloße Ceremonien, sondern für Gnadenmittel hält, durch "
    "welche der heilige Geist wirket. Wer die Taufe empfängt, der wird wiedergeboren. "
    "Die neueren Theologen werden hierin wieder mehr papistisch gesinnt."
)

FEW_SHOT_PASS1 = json.dumps({
    "summary": (
        "Walther defends Lutheran sacramental theology against two fronts: the Schwärmer "
        "who reduce sacraments to empty ceremony, and any drift toward Roman ex opere operato "
        "teaching. He grounds his defense in the confessional claim that sacraments are means "
        "of grace through which the Holy Spirit works, not mere signs. He closes with a sharp "
        "warning that newer theologians are sliding back toward Rome."
    ),
    "tone": "Assertive, polemical, creedal, urgent",
    "argument_structure": [
        {"step": 1, "description": "Opens with concessive affirmation ('Das ist freilich wahr') before pivoting to the Lutheran distinction"},
        {"step": 2, "description": "States the positive Lutheran thesis: sacraments as Gnadenmittel, not mere Ceremonien"},
        {"step": 3, "description": "Grounds thesis in baptismal regeneration (Wiedergeburt) as concrete example"},
        {"step": 4, "description": "Closes with a warning shot against neo-papist drift among newer theologians"}
    ]
}, ensure_ascii=False, indent=2)

FEW_SHOT_PASS2 = json.dumps({
    "theological_vocabulary": [
        {
            "term": "Sacramente",
            "english": "Sacraments",
            "lutheran_nuance": "For Luther: Baptism and Lord's Supper only; efficacious signs that deliver what they signify when received in faith",
            "translation_note": "Term generally matches Catholic or Orthodox usage rather than other Protestant 'ordinances'"
        },
        {
            "term": "bloße Ceremonien",
            "english": "Mere ceremonies / empty rites",
            "lutheran_nuance": "The Zwinglian/Baptist position Luther rejects; sacraments as purely symbolic with no ex opere efficacy",
            "translation_note": "Bloß is emphatic: make sure the translation retains the force."
        },
        {
            "term": "Gnadenmittel",
            "english": "Means of grace",
            "lutheran_nuance": "The Word and Sacraments as objective instruments through which God conveys forgiveness, life, and salvation",
            "translation_note": "Render in the preferred fixed Lutheran technical term in the target language - do not paraphrase"
        },
        {
            "term": "wiedergeboren",
            "english": "Born again / regenerated",
            "lutheran_nuance": "Baptismal regeneration; Luther ties new birth directly to the sacramental act, against spiritualist interiorization",
            "translation_note": "'Born again' - do not soften to 'renewed'"
        },
        {
            "term": "papistisch gesinnt",
            "english": "Disposed toward papism",
            "lutheran_nuance": "Ironic given context: newer Lutherans drifting toward ex opere operato sacramentalism, the Roman error Luther opposed",
            "translation_note": "Emphasize ideological drift, not sudden conversion"
        }
    ]
}, ensure_ascii=False, indent=2)

FEW_SHOT_PASS3 = json.dumps({
    "other_vocabulary": [
        {
            "term": "freilich",
            "english": "To be sure / certainly (concessive)",
            "nuance": "Archaic concessive adverb; signals a controlled admission before a counter-argument, not simple agreement",
            "translation_note": "Preserve the concessive rhetorical function"
        },
        {
            "term": "neueren Theologen",
            "english": "The newer theologians",
            "nuance": "Polemical periodization: 'newer' implies decline from the Reformation standard, not neutral chronology",
            "translation_note": "The pejorative drift must be legible in context"
        }
    ]
}, ensure_ascii=False, indent=2)

FEW_SHOT_PASS4 = json.dumps({
    "rhetorical_devices": [
        {
            "german_phrase": "Die lutherische Kirche hebt die Sacramente überaus hoch",
            "device_type": "word-picture",
            "english_explanation": "Lifting/raising as physical gesture of honor; the church elevates sacraments like a priest elevates the host",
            "sixteenth_century_context": "Elevation of the host in the Mass was a contested liturgical act; Luther's language consciously mirrors and redirects that gesture toward confessional assertion",
            "translation_imperative": "Preserve the physicality of 'lifting' — do not abstract to 'esteems highly'"
        },
        {
            "german_phrase": "Die neueren Theologen werden hierin wieder mehr papistisch gesinnt",
            "device_type": "irony",
            "english_explanation": "Accusing Protestant theologians of the very Roman error Lutheranism was founded to oppose",
            "sixteenth_century_context": "In confessional polemics, 'papistisch' was among the most damning charges; applying it to fellow Lutherans is a sharp rhetorical weapon",
            "translation_imperative": "The ironic sting must be preserved; do not soften to 'tending toward Catholic views'"
        }
    ]
}, ensure_ascii=False, indent=2)

# Map pass number -> (system prompt, few-shot assistant response)
PASSES = {
    1: (SYSTEM_PASS1, FEW_SHOT_PASS1),
    2: (SYSTEM_PASS2, FEW_SHOT_PASS2),
    3: (SYSTEM_PASS3, FEW_SHOT_PASS3),
    4: (SYSTEM_PASS4, FEW_SHOT_PASS4),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_relevant_glossary(text: str) -> str:
    """Return glossary entries whose roots appear in the text."""
    relevant = []
    for de_word, data in GLOSSARY.items():
        translation = data.get("translation")
        root = data.get("root", de_word)
        if not data.get("root") and len(root) > 5:
            root = re.sub(r'(en|er|es|em|e)$', '', root)
        pattern = rf'\b({re.escape(root)})[a-zäöüß]*'
        if re.search(pattern, text, re.IGNORECASE):
            relevant.append(f"{de_word} = {translation}")
    return "\n".join(relevant) if relevant else "Nincs releváns glosszárium."


def run_pass(pass_num: int, chunk_text: str, glossary_str: str) -> dict:
    """Execute a single annotation pass and return parsed JSON."""
    system_prompt, few_shot_assistant = PASSES[pass_num]

    user_content = (
        f"GLOSSZÁRIUM (relevant terms for this passage):\n{glossary_str}\n\n"
        f"PASSAGE:\n{chunk_text}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": FEW_SHOT_USER},
            {"role": "assistant", "content": few_shot_assistant},
            {"role": "user",      "content": user_content},
        ],
        temperature=TEMPERATURE,
        max_tokens=4096,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": PASS_THINKING[pass_num],
                "thinking_token_budget": THINKING_BUDGET,
            }
        },
        timeout=TIMEOUT,
    )

    raw = response.choices[0].message.content.strip()

    # Strip thinking block if present
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    # Strip accidental markdown fences
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$',          '', raw, flags=re.MULTILINE)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = json_repair.repair_json(raw)
        return json.loads(repaired)

# ---------------------------------------------------------------------------
# Canonical output schema
# ---------------------------------------------------------------------------

# Maps each pass to:
#   canonical_key  : the fixed key written to the output JSON
#   candidate_keys : model key variants we will accept (most likely deviations)
#   expected_type  : the Python type we expect the value to be
PASS_SCHEMA = {
    1: {
        "fields": {
            "summary":            (str,  ["summary"]),
            "tone":               (str,  ["tone"]),
            "argument_structure": (list, ["argument_structure", "argumentStructure",
                                          "argument-structure", "argumentstructure"]),
        }
    },
    2: {
        "fields": {
            "theological_vocabulary": (list, ["theological_vocabulary", "theologicalVocabulary",
                                              "theological-vocabulary", "theologicalvocabulary",
                                              "theology_vocabulary", "theol_vocabulary"]),
        }
    },
    3: {
        "fields": {
            "other_vocabulary": (list, ["other_vocabulary", "otherVocabulary",
                                        "other-vocabulary", "othervocabulary",
                                        "vocabulary", "other_vocab"]),
        }
    },
    4: {
        "fields": {
            "rhetorical_devices": (list, ["rhetorical_devices", "rhetoricalDevices",
                                          "rhetorical-devices", "rhetoricaldevices",
                                          "devices", "rhetoric"]),
        }
    },
}


def _to_snake(s: str) -> str:
    """Convert camelCase or hyphen-case to snake_case."""
    s = re.sub(r'([A-Z])', r'_\1', s).lower()
    s = s.replace('-', '_')
    return s.strip('_')


def extract_pass_result(pass_num: int, raw_result: dict) -> dict:
    """
    Extract fields from a model response into a dict with canonical key names.
    Falls back through candidate key aliases; logs a warning if nothing matches.
    """
    schema = PASS_SCHEMA[pass_num]["fields"]
    out = {}

    # Build a snake_case-normalised view of what the model actually returned
    normalised_model = {_to_snake(k): v for k, v in raw_result.items()}

    for canonical_key, (expected_type, candidates) in schema.items():
        value = None

        # 1. Try exact candidate matches first (in priority order)
        for alias in candidates:
            if alias in raw_result:
                value = raw_result[alias]
                break

        # 2. Fall back to snake_case-normalised lookup
        if value is None:
            snake_canonical = _to_snake(canonical_key)
            value = normalised_model.get(snake_canonical)

        # 3. Last resort: any key whose snake_case matches
        if value is None:
            for model_key, model_val in normalised_model.items():
                if _to_snake(model_key) == _to_snake(canonical_key):
                    value = model_val
                    break

        if value is None:
            print(f"  [schema] Pass {pass_num}: '{canonical_key}' not found in model output "
                  f"(keys seen: {list(raw_result.keys())})")
            out[canonical_key] = [] if expected_type is list else ""
        elif not isinstance(value, expected_type):
            print(f"  [schema] Pass {pass_num}: '{canonical_key}' has unexpected type "
                  f"{type(value).__name__}, expected {expected_type.__name__}")
            out[canonical_key] = value  # store as-is; don't silently discard
        else:
            out[canonical_key] = value

    return out


def annotate_chunk(chunk: dict) -> bool:
    chunk_id    = chunk["chunk_id"]
    chunk_text  = chunk["text"].strip()
    output_path = os.path.join(OUTPUT_DIR, f"{chunk_id}.json")

    if os.path.exists(output_path):
        return True  # Resume logic

    glossary_str = get_relevant_glossary(chunk_text)

    combined = {
        "chunk":   chunk_id,
        "lecture": chunk.get("lecture"),
        "thesis":  chunk.get("thesis"),
    }

    for pass_num in range(1, 5):
        try:
            raw_result = run_pass(pass_num, chunk_text, glossary_str)
            combined.update(extract_pass_result(pass_num, raw_result))
        except json.JSONDecodeError as e:
            print(f"\n[{chunk_id}] Pass {pass_num} JSON parse error: {e}")
            combined[f"pass{pass_num}_error"] = str(e)
        except Exception as e:
            print(f"\n[{chunk_id}] Pass {pass_num} error: {e}")
            combined[f"pass{pass_num}_error"] = str(e)
            return False

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Four-pass catechism annotation pipeline")
    parser.add_argument("start",        nargs="?", type=int, default=1,    help="First chunk index (1-based)")
    parser.add_argument("end",          nargs="?", type=int, default=9999, help="Last chunk index (inclusive)")
    parser.add_argument("--batch-size", type=int,  default=4,              help="Concurrent chunks (default: 4)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    # Sort by chunk_id numerically, then slice
    all_chunks.sort(key=lambda c: int(c["chunk_id"]))
    target_chunks = all_chunks[args.start - 1 : args.end]

    print(f"Pipeline start: chunks {args.start}–{args.end} "
          f"({len(target_chunks)} chunks, batch_size={args.batch_size})")
    print(f"Model: {MODEL} | Temp: {TEMPERATURE} | Thinking budget: {THINKING_BUDGET} tok\n")

    errors = []

    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {executor.submit(annotate_chunk, c): c["chunk_id"] for c in target_chunks}

        with tqdm(total=len(target_chunks), unit="chunk") as pbar:
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    ok = future.result()
                    if not ok:
                        errors.append(cid)
                except Exception as e:
                    print(f"\nUnhandled exception on {cid}: {e}")
                    errors.append(cid)
                pbar.update(1)

    if errors:
        print(f"\nCompleted with {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
    else:
        print("\nAll chunks completed successfully.")


if __name__ == "__main__":
    main()
