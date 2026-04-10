"""
translator.py - Step 4: Translation with Full Context

This is the core translation engine. It takes a chunk along with all the
context we've built (glossary, surrounding chunks, document structure)
and produces a high-quality theological translation.

WHY THIS IS DIFFERENT FROM "JUST TRANSLATING":
  If you simply ask an AI "translate this German text to Hungarian," you'll
  get a passable translation. But for doctrinal texts, passable isn't enough.
  The difference between a good and great translation prompt is:

  1. CONTEXT: The AI knows what document this is from and what theological
     topic is being discussed.
  2. GLOSSARY: The AI uses approved translations for key terms instead of
     making up its own each time.
  3. SURROUNDING TEXT: The AI sees what came before and after, so it can
     maintain coherence across chunk boundaries.
  4. EXPLICIT INSTRUCTIONS: The AI is told to prioritize doctrinal precision
     over natural-sounding language.
  5. PRESERVATION RULES: The AI is told not to "improve" the text — to keep
     the author's style, repetitions, and rhetorical force.

INPUT:  A chunk dict (from chunker.py) + glossary (from glossary.py)
OUTPUT: The translated text + metadata about the translation
"""

import json
import os
import sys
from anthropic import Anthropic

sys.path.insert(0, os.path.dirname(__file__))
from config import TRANSLATE_MODEL, MAX_TOKENS, SOURCE_LANGUAGE


def get_client():
    return Anthropic()


# =========================================================================
# TRANSLATION SYSTEM PROMPT
# =========================================================================

def build_system_prompt(source_language, target_language, document_context=None):
    """
    Builds the system prompt for the translator. This is where all of our
    doctrinal accuracy instructions live.
    """
    context_section = ""
    if document_context:
        context_section = f"""
DOCUMENT CONTEXT:
{document_context}
"""

    return f"""You are an expert translator of Lutheran confessional documents.
You are translating from {source_language} to {target_language} for the
Wisconsin Evangelical Lutheran Synod (WELS).

YOUR MISSION: Produce a translation that is doctrinally perfect. A translation
that sounds beautiful but subtly shifts a doctrinal meaning is WORSE than
one that sounds slightly awkward but preserves the theology perfectly.
{context_section}
ABSOLUTE RULES:

1. DOCTRINAL PRECISION IS NON-NEGOTIABLE.
   Every theological concept must be translated with its exact meaning.
   Do not soften, generalize, or modernize doctrinal language.
   "Justification" means forensic declaration of righteousness — not
   "being made right" or "finding peace with God."

2. USE THE GLOSSARY TERMS EXACTLY.
   When a glossary is provided, you MUST use the specified translations
   for those terms. Do not substitute synonyms or alternatives.
   The glossary exists to ensure consistency across the entire document.

3. PRESERVE THE AUTHOR'S STYLE.
   Do not "improve" the text. If the original is repetitive, keep it
   repetitive. If the original is harsh (especially in law passages),
   keep it harsh. The rhetorical force is part of the theology.
   Do not break long sentences into shorter ones unless absolutely
   necessary for comprehension in {target_language}.

4. PRESERVE STRUCTURE.
   Maintain paragraph breaks, numbered points, and section markers
   exactly as they appear in the source.

5. DO NOT ADD OR REMOVE CONTENT.
   Translate what is there. Do not add explanatory notes, do not
   skip passages, do not paraphrase. Every sentence in the source
   must have a corresponding sentence in the translation.

6. SCRIPTURE REFERENCES.
   When the text quotes or cites Scripture, use the standard
   {target_language} Bible translation for that passage if one exists.
   Preserve the citation format (e.g., "Rom. 3:28").

7. LATIN/GREEK PHRASES.
   If the original includes Latin or Greek theological phrases
   (e.g., "sola fide," "simul justus et peccator"), keep them in
   Latin/Greek with a {target_language} translation in parentheses
   on first occurrence only.

OUTPUT FORMAT:
Respond with ONLY the translated text. No explanations, no notes, no
commentary. Just the translation."""


# =========================================================================
# TRANSLATION
# =========================================================================

def translate_chunk(chunk, target_language, glossary=None,
                    source_language=None, document_context=None,
                    quote_instructions=None, previous_translation=None):
    """
    Translates a single chunk with full context.

    Parameters:
        chunk (dict): A chunk from chunker.py with keys:
                      text, context_before, context_after, article, chunk_id
        target_language (str): Language to translate into
        glossary (dict): {source_term: target_translation} for this chunk
        source_language (str): Source language override
        document_context (str): Description of the document and section
        quote_instructions (str): Bible verse translations and quote handling
                                  instructions from quote_handler.py
        previous_translation (str): The translated text from the PREVIOUS chunk.
                                    Used to maintain terminology consistency
                                    across chunks (e.g., always using "Cikkely"
                                    for "Article" instead of sometimes "Cikk").

    Returns:
        dict: {
            "chunk_id": str,
            "source_text": str,
            "translated_text": str,
            "target_language": str,
            "glossary_used": dict,
            "article": str,
        }
    """
    lang = source_language or SOURCE_LANGUAGE
    client = get_client()

    # Build the system prompt with all our doctrinal instructions
    system = build_system_prompt(lang, target_language, document_context)

    # Build the user message with the chunk + context + glossary
    user_parts = []

    # Add glossary if we have one
    if glossary:
        glossary_text = "\n".join(
            f"  {src} = {tgt}" for src, tgt in glossary.items()
        )
        user_parts.append(
            f"REQUIRED GLOSSARY — Use these exact translations for these terms:\n{glossary_text}"
        )

    # Add quote instructions if available (Bible verses, patristic quotes, etc.)
    if quote_instructions:
        user_parts.append(quote_instructions)

    # CONSISTENCY ANCHOR: show the translator what it produced for the
    # previous chunk so it maintains the same terminology, style, and
    # formatting conventions (e.g., "Cikkely" not "Cikk", consistent
    # numbering formats, same register).  We include the tail end of the
    # previous translation — enough for the model to pick up patterns
    # without bloating the prompt.
    if previous_translation:
        # Take the last ~500 chars (roughly the last paragraph or two)
        tail = previous_translation[-500:] if len(previous_translation) > 500 else previous_translation
        # If we sliced mid-sentence, skip to the first sentence boundary
        if len(previous_translation) > 500:
            first_period = tail.find('. ')
            if first_period != -1 and first_period < 100:
                tail = tail[first_period + 2:]
            tail = "..." + tail
        user_parts.append(
            f"YOUR PREVIOUS TRANSLATION (the chunk you just translated — "
            f"maintain the SAME terminology, style, and formatting conventions "
            f"you used here. If you used a specific word for a concept, keep "
            f"using that same word. Do NOT translate this section again):\n{tail}"
        )

    # Add surrounding context if available
    context_before = chunk.get("context_before", "")
    context_after = chunk.get("context_after", "")

    if context_before:
        user_parts.append(
            f"PRECEDING CONTEXT (for reference only, do NOT translate this):\n...{context_before}"
        )

    # The actual text to translate
    user_parts.append(
        f"TEXT TO TRANSLATE (translate ONLY this section):\n{chunk['text']}"
    )

    if context_after:
        user_parts.append(
            f"FOLLOWING CONTEXT (for reference only, do NOT translate this):\n{context_after}..."
        )

    user_message = "\n\n".join(user_parts)

    # Make the API call
    response = client.messages.create(
        model=TRANSLATE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )

    translated_text = response.content[0].text.strip()

    return {
        "chunk_id": chunk.get("chunk_id", "unknown"),
        "source_text": chunk["text"],
        "translated_text": translated_text,
        "target_language": target_language,
        "source_language": lang,
        "glossary_used": glossary or {},
        "article": chunk.get("article"),
        "position": chunk.get("position"),
    }


# =========================================================================
# BATCH TRANSLATION
# =========================================================================

def translate_chunks(chunks, target_language, glossary_path=None,
                     source_language=None, document_context=None,
                     handle_quotes=True):
    """
    Translates a list of chunks, building/using the glossary along the way.

    Parameters:
        chunks (list): List of chunk dicts from chunker.py
        target_language (str): Language to translate into
        glossary_path (str): Path to glossary file (for loading/building)
        source_language (str): Source language override
        document_context (str): Description of the document
        handle_quotes (bool): Whether to detect and resolve Bible/other quotes

    Returns:
        list: List of translation result dicts
    """
    from glossary import build_glossary_for_chunk, load_glossary

    lang = source_language or SOURCE_LANGUAGE
    results = []

    # Track the previous chunk's translation for consistency anchoring.
    # Benjamin noted that without this, chunks independently choose
    # different terms for the same concept (e.g., "Cikk" vs "Cikkely").
    previous_translation = None

    for i, chunk in enumerate(chunks):
        print(f"\n=== Translating chunk {chunk.get('chunk_id', i+1)} ({i+1}/{len(chunks)}) ===")
        article = chunk.get("article", "")
        if article:
            print(f"  Article: {article}")

        # Step 3: Build/lookup glossary for this chunk
        if glossary_path:
            print(f"  Building glossary...")
            chunk_glossary = build_glossary_for_chunk(
                chunk["text"], target_language, glossary_path, lang
            )
            print(f"  Glossary: {len(chunk_glossary)} terms for this chunk")
        else:
            chunk_glossary = None

        # Step 3.5: Detect and resolve quotes (Bible verses, etc.)
        quote_instructions = None
        if handle_quotes:
            from quote_handler import process_quotes_for_chunk
            print(f"  Processing quotes...")
            quote_data = process_quotes_for_chunk(
                chunk["text"], target_language, lang,
                use_ai_detection=True,
            )
            quote_instructions = quote_data.get("quote_instructions", "")
            if quote_data.get("bible_verses"):
                print(f"  Bible verses resolved: {len(quote_data['bible_verses'])}")
            if quote_data.get("other_quotes"):
                print(f"  Other quotes found: {len(quote_data['other_quotes'])}")

        # Step 4: Translate with full context + previous translation for consistency
        if previous_translation:
            print(f"  Consistency anchor: passing {min(500, len(previous_translation))} chars from previous chunk")
        print(f"  Translating {lang} -> {target_language}...")
        result = translate_chunk(
            chunk=chunk,
            target_language=target_language,
            glossary=chunk_glossary,
            source_language=lang,
            document_context=document_context,
            quote_instructions=quote_instructions,
            previous_translation=previous_translation,
        )
        results.append(result)

        # Update the consistency anchor for the next chunk
        previous_translation = result["translated_text"]

        print(f"  Done: {len(result['translated_text'])} chars")

    return results


# =========================================================================
# CLI ENTRY POINT
# =========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Step 4: Translate chunks with full context and glossary"
    )
    parser.add_argument(
        "input", type=str,
        help="Path to a chunks JSON file (from chunker.py)"
    )
    parser.add_argument(
        "--target", type=str, required=True,
        help="Target language (e.g., 'Hungarian', 'Spanish', 'Hmong')"
    )
    parser.add_argument(
        "--glossary", type=str, default=None,
        help="Path to glossary file (built automatically if not provided)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for translations JSON"
    )
    parser.add_argument(
        "--context", type=str, default=None,
        help="Document context description (e.g., 'Article IV of the Augsburg Confession, on Justification')"
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None,
        help="Limit number of chunks to translate (for testing)"
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Source language override (default: from config.py)"
    )
    args = parser.parse_args()

    # Load chunks
    print(f"Loading chunks from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    print(f"  {len(chunks)} chunks to translate")

    # Set up glossary path
    glossary_path = args.glossary or os.path.join(
        os.path.dirname(__file__), "data",
        f"glossary_{args.target.lower()}.json"
    )

    # Translate
    results = translate_chunks(
        chunks=chunks,
        target_language=args.target,
        glossary_path=glossary_path,
        source_language=args.language,
        document_context=args.context,
    )

    # Save results
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "data",
        f"translations_{args.target.lower()}.json"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} translations to {output_path}")


if __name__ == "__main__":
    main()
