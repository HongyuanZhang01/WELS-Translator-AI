"""
glossary.py - Step 3: Theological Glossary Generator

Before translating, this step identifies the key theological terms in each
chunk and establishes consistent translations for them.

WHY THIS MATTERS:
  If "justification" is translated as word A in paragraph 3 and word B in
  paragraph 40, the reader thinks two different concepts are being discussed.
  Worse, if a key term like "means of grace" gets translated as a generic
  phrase instead of the standard theological vocabulary, the doctrinal
  precision is lost.

HOW IT WORKS:
  1. Scans a chunk (or batch of chunks) for theological terms
  2. Checks if we already have approved translations for those terms
  3. For new terms, asks an AI to propose a translation WITH an explanation
     of why that word was chosen
  4. Saves everything to a growing glossary file

  The glossary is cumulative — it grows as you process more documents.
  Once a term is in the glossary, it's used consistently forever.

INPUT:  Source text chunks + target language
OUTPUT: A glossary JSON file mapping source terms to target terms
"""

import json
import os
import sys
from anthropic import Anthropic

sys.path.insert(0, os.path.dirname(__file__))
from config import EVAL_MODEL, MAX_TOKENS, SOURCE_LANGUAGE


def get_client():
    return Anthropic()


# =========================================================================
# TERM EXTRACTION
# =========================================================================

EXTRACTION_PROMPT = """PIPELINE CONTEXT: You are Step 2 (Glossary Building) in a WELS Lutheran
translation pipeline. Before you, the source document was chunked into passages. Your job
is to identify theological terms that need consistent translation. After you, another AI
will translate each term, and then Step 4 (Translation) will use your glossary to ensure
every chunk uses the same terminology.

YOUR ROLE: You are ONLY extracting terms. You are NOT translating them — that happens
in a separate step. You are NOT evaluating or improving anything. Just identify the terms
that matter doctrinally.

You are an expert in Lutheran theological terminology,
particularly the Book of Concord and WELS confessional documents.

Given a text in {source_language}, identify ALL theologically significant terms
and phrases. These include:

- Core doctrinal concepts (justification, sanctification, means of grace, etc.)
- Lutheran-specific terminology (sola fide, sola gratia, simul justus et peccator, etc.)
- Biblical/theological proper nouns with doctrinal weight (Sacrament, Baptism, Lord's Supper, etc.)
- Terms where a generic translation would lose doctrinal precision
- Key confessional phrases that carry specific meaning in Lutheran theology

Do NOT include:
- Common words that have no special theological meaning
- Names of people or places (unless they have doctrinal significance)
- Generic connecting words or phrases

Respond with valid JSON only (no markdown, no code blocks):
{{
  "terms": [
    {{
      "term": "<the term in {source_language}>",
      "doctrinal_significance": "<brief explanation of why this term matters doctrinally>"
    }}
  ]
}}"""


def extract_terms(text, source_language=None):
    """
    Uses AI to identify theologically significant terms in a text.

    Parameters:
        text (str): The source text to scan
        source_language (str): The language of the text

    Returns:
        list: List of dicts with "term" and "doctrinal_significance"
    """
    lang = source_language or SOURCE_LANGUAGE
    client = get_client()

    response = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=MAX_TOKENS,
        system=EXTRACTION_PROMPT.format(source_language=lang),
        messages=[{
            "role": "user",
            "content": f"Identify all theologically significant terms in this {lang} text:\n\n{text}"
        }],
    )

    response_text = response.content[0].text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

    try:
        result = json.loads(response_text)
        return result.get("terms", [])
    except json.JSONDecodeError:
        print(f"  WARNING: Failed to parse term extraction response")
        return []


# =========================================================================
# TERM TRANSLATION
# =========================================================================

TRANSLATION_PROMPT = """PIPELINE CONTEXT: You are Step 3 (Glossary Translation) in a WELS
Lutheran translation pipeline. Step 2 identified theologically significant terms in the
source text. Your job is to translate each term into the target language. After you, Step 4
(Translation) will use your glossary to translate full passages — every time a glossary
term appears, the translator MUST use your exact translation. This means your term choices
will be enforced throughout the entire document. Choose carefully.

YOUR ROLE: You are ONLY translating individual terms. You are NOT translating full
passages — that happens in Step 4. You are NOT evaluating anything. Your output is a
glossary that the translator will follow as mandatory.

WHAT NOT TO DO:
- Do NOT provide multiple acceptable translations — pick ONE definitive translation per term.
- Do NOT use generic words when a specific theological term exists in {target_language}.
- Do NOT invent new terms if an established one exists in Lutheran {target_language} literature.

You are an expert translator specializing in Lutheran theological terminology.
You are translating terms from {source_language} to {target_language} for use in WELS
(Wisconsin Evangelical Lutheran Synod) confessional documents.

CRITICAL RULES:
1. Use the STANDARD theological vocabulary in {target_language} whenever
   one exists. Do not invent new translations for established terms.
2. If multiple translations exist, prefer the one used in established
   {target_language} Lutheran publications or Bible translations.
3. If no standard translation exists (especially for low-resource languages),
   create one that precisely captures the doctrinal meaning, and explain
   your reasoning thoroughly.
4. NEVER use a generic word when a specific theological term exists.

For each term, provide:
- The recommended translation
- Why you chose this translation
- Any alternatives that were considered and why they were rejected

Respond with valid JSON only (no markdown, no code blocks):
{{
  "translations": [
    {{
      "source_term": "<term in {source_language}>",
      "translation": "<recommended translation in {target_language}>",
      "reasoning": "<why this translation was chosen>",
      "alternatives": ["<other options considered>"],
      "confidence": "<high|medium|low>"
    }}
  ]
}}"""


def translate_terms(terms, target_language, source_language=None):
    """
    Translates a batch of theological terms into the target language.

    Parameters:
        terms (list): List of term dicts from extract_terms()
        target_language (str): The language to translate into
        source_language (str): The language of the source terms

    Returns:
        list: List of translation dicts with source, target, reasoning
    """
    lang = source_language or SOURCE_LANGUAGE
    client = get_client()

    # Format terms for the prompt
    terms_text = "\n".join(
        f"- {t['term']}: {t.get('doctrinal_significance', '')}"
        for t in terms
    )

    response = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=MAX_TOKENS,
        system=TRANSLATION_PROMPT.format(
            source_language=lang,
            target_language=target_language,
        ),
        messages=[{
            "role": "user",
            "content": f"Translate these {lang} theological terms to {target_language}:\n\n{terms_text}"
        }],
    )

    response_text = response.content[0].text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

    try:
        result = json.loads(response_text)
        return result.get("translations", [])
    except json.JSONDecodeError:
        print(f"  WARNING: Failed to parse term translation response")
        return []


# =========================================================================
# GLOSSARY MANAGEMENT
# =========================================================================

def load_glossary(glossary_path):
    """Load an existing glossary from disk, or return empty if none exists."""
    if os.path.exists(glossary_path):
        with open(glossary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_glossary(glossary, glossary_path):
    """Save the glossary to disk."""
    os.makedirs(os.path.dirname(glossary_path) or ".", exist_ok=True)
    with open(glossary_path, "w", encoding="utf-8") as f:
        json.dump(glossary, f, indent=2, ensure_ascii=False)
    print(f"  Glossary saved: {len(glossary)} terms -> {glossary_path}")


def build_glossary_for_chunk(chunk_text, target_language, glossary_path,
                              source_language=None):
    """
    The main function for Step 3. For a given chunk:
    1. Extract theological terms
    2. Check which ones are already in the glossary
    3. Translate any new ones
    4. Add them to the glossary
    5. Return the glossary subset relevant to this chunk

    Parameters:
        chunk_text (str): The source text of this chunk
        target_language (str): Target language for translations
        glossary_path (str): Path to the glossary JSON file
        source_language (str): Source language override

    Returns:
        dict: {source_term: target_translation} for terms in this chunk
    """
    lang = source_language or SOURCE_LANGUAGE

    # Load existing glossary
    full_glossary = load_glossary(glossary_path)

    # Extract terms from this chunk
    print(f"  Extracting theological terms...")
    terms = extract_terms(chunk_text, lang)
    print(f"  Found {len(terms)} theological terms")

    # Find which terms are NOT yet in the glossary
    new_terms = [
        t for t in terms
        if t["term"].lower() not in {k.lower() for k in full_glossary}
    ]

    if new_terms:
        print(f"  {len(new_terms)} new terms need translation...")
        translations = translate_terms(new_terms, target_language, lang)

        # Add new translations to the glossary
        for t in translations:
            source = t.get("source_term", "")
            target = t.get("translation", "")
            if source and target:
                full_glossary[source] = {
                    "translation": target,
                    "reasoning": t.get("reasoning", ""),
                    "alternatives": t.get("alternatives", []),
                    "confidence": t.get("confidence", "unknown"),
                    "target_language": target_language,
                }

        save_glossary(full_glossary, glossary_path)
    else:
        print(f"  All terms already in glossary")

    # Return just the simple {source: target} mapping for this chunk's terms
    chunk_glossary = {}
    for t in terms:
        term = t["term"]
        for key in full_glossary:
            if key.lower() == term.lower():
                entry = full_glossary[key]
                if isinstance(entry, dict):
                    chunk_glossary[term] = entry["translation"]
                else:
                    chunk_glossary[term] = entry
                break

    return chunk_glossary


# =========================================================================
# CLI ENTRY POINT
# =========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Step 3: Build a theological glossary from source text"
    )
    parser.add_argument(
        "input", type=str,
        help="Path to a chunks JSON file or a plain text file"
    )
    parser.add_argument(
        "--target", type=str, required=True,
        help="Target language (e.g., 'Hungarian', 'Spanish', 'Hmong')"
    )
    parser.add_argument(
        "--glossary", type=str, default=None,
        help="Path to glossary file (created if doesn't exist)"
    )
    parser.add_argument(
        "--max-chunks", type=int, default=5,
        help="Maximum number of chunks to process (default: 5)"
    )
    args = parser.parse_args()

    # Load input
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        chunks = data[:args.max_chunks]
    else:
        chunks = [data]

    glossary_path = args.glossary or os.path.join(
        os.path.dirname(__file__), "data",
        f"glossary_{args.target.lower()}.json"
    )

    print(f"Building glossary for {args.target}...")
    print(f"Processing {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", chunk) if isinstance(chunk, dict) else chunk
        chunk_id = chunk.get("chunk_id", f"{i+1:03d}") if isinstance(chunk, dict) else f"{i+1:03d}"
        print(f"\n--- Chunk {chunk_id} ---")
        build_glossary_for_chunk(text, args.target, glossary_path)

    final = load_glossary(glossary_path)
    print(f"\nFinal glossary: {len(final)} terms for {args.target}")


if __name__ == "__main__":
    main()
