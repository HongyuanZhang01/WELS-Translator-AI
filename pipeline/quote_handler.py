"""
quote_handler.py - Detect and resolve quotes (especially Bible references)

WHY THIS EXISTS:
  The Augsburg Confession (and all confessional documents) are full of Bible
  quotes. When we translate "For by grace are ye saved through faith" (Eph 2:8)
  into Hungarian, we should NOT have the AI translate the English quote.
  Instead, we should use the ACTUAL published Hungarian Bible text for Eph 2:8.

  Why? Because:
  1. Published Bible translations have been carefully reviewed by theologians
     for decades. Our AI translation of a Bible verse will never be as trusted.
  2. Native speakers RECOGNIZE their Bible. If a Hungarian pastor reads our
     translation and the Bible quotes don't match the Karoli Bible (the
     standard Hungarian Protestant Bible), they'll immediately distrust
     the entire translation.
  3. Consistency: every time Eph 2:8 appears in any document, it should
     use the exact same Hungarian words.

HOW IT WORKS:
  1. DETECT: Scan the source text for Bible references (e.g., "Rom. 3:28",
     "John 14, 6", "1 Cor. 9, 27") and extract both the reference and the
     quoted text.
  2. RESOLVE: Look up the actual published translation for each reference
     in the target language, using Bible APIs or known translations.
  3. INJECT: Provide the translator with a mapping of {reference: real_quote}
     so it uses the published text instead of generating its own.

  This also handles non-Biblical quotes (e.g., quotes from Church Fathers)
  by flagging them for the translator to handle with extra care.

INPUT:  Source text chunk + target language
OUTPUT: A dict mapping Bible references to their real translations
"""

import re
import json
import os
import sys
from anthropic import Anthropic

sys.path.insert(0, os.path.dirname(__file__))
from config import EVAL_MODEL, MAX_TOKENS, SOURCE_LANGUAGE


def get_client():
    return Anthropic()


# =========================================================================
# HUMAN REVIEW QUEUE — for Bible lookups the API refused to resolve
# =========================================================================
#
# When the Anthropic API returns a 4xx/5xx on a Bible verse lookup (e.g.
# a content-filter false positive on a Communion or covenant verse, a
# transient server error, or a rate limit), we can't crash the whole run
# over one bad lookup. Instead, we record the references and the error
# so a human translator/reviewer can verify the AI's inline rendering of
# those verses against the canonical target-language Bible after the run.
#
# The translator still produces output — it just doesn't get a pre-resolved
# canonical verse text for those specific references, so the model falls
# back on its own knowledge. That's why these entries need human review.

BIBLE_LOOKUP_FAILURES = []


def get_bible_lookup_failures():
    """Return a copy of the current human-review queue (list of dicts)."""
    return list(BIBLE_LOOKUP_FAILURES)


def clear_bible_lookup_failures():
    """Reset the human-review queue. Call at the start of each run."""
    BIBLE_LOOKUP_FAILURES.clear()


def _record_bible_lookup_failure(chunk_id, references, target_language,
                                 error_type, error_message):
    """Append a failure record for later human review."""
    BIBLE_LOOKUP_FAILURES.append({
        "chunk_id": chunk_id,
        "target_language": target_language,
        "references": list(references),
        "error_type": error_type,
        "error_message": error_message,
        "reason": (
            "The Anthropic API refused to return canonical Bible text for "
            "these references. The translator produced its own rendering "
            "inline — a human must verify it against the standard published "
            f"{target_language} Bible before delivery."
        ),
    })


# =========================================================================
# BIBLE REFERENCE DETECTION
# =========================================================================

# Common abbreviations for Bible books (English and Latin/German forms)
BIBLE_BOOKS = {
    # Old Testament
    "Gen": "Genesis", "Ex": "Exodus", "Lev": "Leviticus", "Num": "Numbers",
    "Deut": "Deuteronomy", "Josh": "Joshua", "Judg": "Judges", "Ruth": "Ruth",
    "1 Sam": "1 Samuel", "2 Sam": "2 Samuel", "1 Kings": "1 Kings",
    "2 Kings": "2 Kings", "1 Chron": "1 Chronicles", "2 Chron": "2 Chronicles",
    "Ezra": "Ezra", "Neh": "Nehemiah", "Esth": "Esther", "Job": "Job",
    "Ps": "Psalms", "Prov": "Proverbs", "Eccl": "Ecclesiastes",
    "Song": "Song of Solomon", "Isa": "Isaiah", "Jer": "Jeremiah",
    "Lam": "Lamentations", "Ezek": "Ezekiel", "Dan": "Daniel",
    "Hos": "Hosea", "Joel": "Joel", "Amos": "Amos", "Obad": "Obadiah",
    "Jonah": "Jonah", "Mic": "Micah", "Nah": "Nahum", "Hab": "Habakkuk",
    "Zeph": "Zephaniah", "Hag": "Haggai", "Zech": "Zechariah", "Mal": "Malachi",
    # New Testament
    "Matt": "Matthew", "Mark": "Mark", "Luke": "Luke", "John": "John",
    "Acts": "Acts", "Rom": "Romans", "1 Cor": "1 Corinthians",
    "2 Cor": "2 Corinthians", "Gal": "Galatians", "Eph": "Ephesians",
    "Phil": "Philippians", "Col": "Colossians", "1 Thess": "1 Thessalonians",
    "2 Thess": "2 Thessalonians", "1 Tim": "1 Timothy", "2 Tim": "2 Timothy",
    "Titus": "Titus", "Philem": "Philemon", "Heb": "Hebrews",
    "James": "James", "1 Pet": "1 Peter", "2 Pet": "2 Peter",
    "1 John": "1 John", "2 John": "2 John", "3 John": "3 John",
    "Jude": "Jude", "Rev": "Revelation",
    # German abbreviations
    "Röm": "Romans", "Kor": "1 Corinthians", "Joh": "John",
    "Matth": "Matthew", "Luk": "Luke", "Apg": "Acts",
    "1 Kor": "1 Corinthians", "2 Kor": "2 Corinthians",
}


def detect_bible_references(text):
    """
    Finds Bible references in the source text using pattern matching.

    Handles formats like:
      - "Rom. 3, 28" (German style with comma)
      - "Rom. 3:28" (English style with colon)
      - "1 Cor. 9, 27" (with book number)
      - "John 14, 6" (full book name)
      - "Ps. 119, 46" (abbreviated)
      - "Matt. 17, 21" (abbreviated)

    Returns:
        list: List of dicts with reference info:
              {"raw": "Rom. 3, 28", "book": "Romans", "chapter": "3",
               "verse": "28", "standardized": "Romans 3:28"}
    """
    # Pattern: optional number + book name/abbrev + period optional + chapter + separator + verse(s)
    pattern = r'(?:(\d)\s*)?([A-Za-zÖöÜüÄä]+)\.?\s+(\d{1,3})\s*[,:]\s*(\d{1,3}(?:\s*[-–]\s*\d{1,3})?)'

    matches = []
    for m in re.finditer(pattern, text):
        book_num = m.group(1) or ""
        book_abbrev = m.group(2)
        chapter = m.group(3)
        verse = m.group(4).replace(" ", "")

        # Try to resolve the book name
        full_key = f"{book_num} {book_abbrev}".strip() if book_num else book_abbrev

        book_name = None
        for abbrev, name in BIBLE_BOOKS.items():
            if full_key.lower() == abbrev.lower() or book_abbrev.lower() == abbrev.lower():
                book_name = name
                if book_num:
                    # Check if the number should be part of the book name
                    numbered_key = f"{book_num} {abbrev}"
                    if numbered_key in BIBLE_BOOKS:
                        book_name = BIBLE_BOOKS[numbered_key]
                break

        # Also check full book names
        if not book_name:
            for abbrev, name in BIBLE_BOOKS.items():
                if book_abbrev.lower().startswith(abbrev.lower()):
                    book_name = name
                    break

        if book_name:
            if book_num and not book_name[0].isdigit():
                book_name = f"{book_num} {book_name}"

            matches.append({
                "raw": m.group(0),
                "book": book_name,
                "chapter": chapter,
                "verse": verse,
                "standardized": f"{book_name} {chapter}:{verse}",
                "position": m.start(),
            })

    return matches


# =========================================================================
# AI-BASED QUOTE DETECTION (catches what regex misses)
# =========================================================================

QUOTE_DETECTION_PROMPT = """PIPELINE CONTEXT: You are Step 3.5 (Quote Detection) in a WELS
Lutheran translation pipeline. Before you, the source text was chunked and a glossary was
built. Your job is to find every quote in the source text so that the translator (Step 4)
can use the official published translation of each quote instead of translating it fresh.
This is critical — Bible verses and creedal quotes must match the standard published
translation in the target language, not be independently retranslated.

YOUR ROLE: You are ONLY detecting quotes. You are NOT translating them — that happens
in a separate lookup step. You are NOT evaluating anything. Just find the quotes and
identify their sources accurately.

WHAT NOT TO DO:
- Do NOT translate any quotes.
- Do NOT skip quotes because they seem unimportant — let the pipeline decide importance.
- Do NOT fabricate references. If you're unsure of a source, say "uncertain" rather than guess.

You are an expert in Biblical and theological texts.
Analyze the following text and identify ALL quotes, including:

1. BIBLE QUOTES: Direct quotations from Scripture, with their references.
2. PATRISTIC QUOTES: Quotes from Church Fathers (Augustine, Ambrose, etc.)
3. CREEDAL QUOTES: Quotes from creeds or other confessional documents.
4. OTHER QUOTES: Any other quoted material.

For each quote, identify:
- The quoted text (as it appears in the source)
- The source/reference (e.g., "Romans 3:28" or "Augustine, De Vocatione Gentium")
- The type (bible, patristic, creedal, other)

Respond with valid JSON only (no markdown, no code blocks):
{{
  "quotes": [
    {{
      "quoted_text": "<the actual quoted text from the source>",
      "reference": "<the citation/source>",
      "type": "<bible|patristic|creedal|other>",
      "standardized_ref": "<standardized reference, e.g., 'Romans 3:28'>"
    }}
  ]
}}"""


def detect_all_quotes(text, source_language=None):
    """
    Uses AI to find ALL quotes in the text — Bible, Church Fathers, creeds, etc.
    This catches quotes that regex might miss (unusual formatting, implied references).

    Returns:
        list: List of quote dicts with text, reference, type
    """
    lang = source_language or SOURCE_LANGUAGE
    client = get_client()

    response = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=MAX_TOKENS,
        system=QUOTE_DETECTION_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Find all quotes in this {lang} text:\n\n{text}"
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
        return result.get("quotes", [])
    except json.JSONDecodeError:
        print(f"  WARNING: Failed to parse quote detection response")
        return []


# =========================================================================
# BIBLE VERSE LOOKUP
# =========================================================================

# Standard Bible translations by language
# These are the "default" Bibles that native-speaking pastors would recognize
STANDARD_BIBLES = {
    "Hungarian": "Karoli Bible (Karoli Gaspar, revised)",
    "Spanish": "Reina-Valera 1960",
    "Hmong": "Hmong Daw Bible (Vajtswv Txojlus)",
    "German": "Luther Bible 2017",
    "French": "Louis Segond 1910",
    "Swahili": "Biblia Habari Njema",
    "Indonesian": "Terjemahan Baru (TB)",
    "Russian": "Synodal Translation",
    "Mandarin": "Chinese Union Version (CUV)",
    "Portuguese": "Almeida Revista e Atualizada",
}

BIBLE_LOOKUP_PROMPT = """PIPELINE CONTEXT: You are Step 3.5b (Bible Verse Lookup) in a WELS
Lutheran translation pipeline. Step 3.5a detected Bible verse references in the source
text. Your job is to provide the EXACT published text of those verses in the target
language so the translator (Step 4) can embed the official wording instead of translating
Scripture independently. This ensures doctrinal fidelity — a pastor reading the translation
must see the same Bible text they would see in their own Bible.

YOUR ROLE: You are ONLY a Bible text lookup tool. You are NOT translating verses yourself.
You are recalling the published text from a specific Bible translation.

WHAT NOT TO DO:
- Do NOT translate the verse yourself from Hebrew/Greek/English. Recall the PUBLISHED text.
- Do NOT modify, modernize, or "improve" the published wording.
- Do NOT provide a verse from the wrong Bible translation.
- If you cannot recall the exact wording, mark it "UNCERTAIN" — do NOT guess.

You are a Bible translation expert. You have memorized the standard published
Bible translations in many languages.

I need the EXACT text of specific Bible verses from the {bible_name}
({target_language} standard Bible translation).

CRITICAL: You must provide the ACTUAL published text from this specific
translation, NOT your own translation of the verse. If you are unsure of
the exact wording in this specific translation, say "UNCERTAIN" and provide
your best approximation with a note.

For each verse, provide:
- The exact text as it appears in the {bible_name}
- Your confidence level (certain/likely/uncertain)

Respond with valid JSON only (no markdown, no code blocks):
{{
  "verses": [
    {{
      "reference": "<standardized reference>",
      "text": "<exact text from the {bible_name}>",
      "confidence": "<certain|likely|uncertain>"
    }}
  ]
}}"""


def lookup_bible_verses(references, target_language, chunk_id=None):
    """
    Looks up Bible verses in the standard published Bible translation
    for the target language.

    Parameters:
        references (list): List of standardized references (e.g., ["Romans 3:28", "John 14:6"])
        target_language (str): The target language
        chunk_id: Optional chunk identifier, recorded if the lookup fails
                  so a human reviewer can find the right chunk to check.

    Returns:
        dict: {reference: {"text": verse_text, "confidence": level, "bible": bible_name}}

        On any API error (content-filter block, rate limit, server error,
        network failure) this returns an EMPTY DICT and records the failure
        in the module-level BIBLE_LOOKUP_FAILURES queue for human review
        instead of raising. This keeps one bad lookup from killing the run.
    """
    if not references:
        return {}

    bible_name = STANDARD_BIBLES.get(target_language, f"standard {target_language} Bible")
    client = get_client()

    refs_text = "\n".join(f"- {ref}" for ref in references)

    try:
        response = client.messages.create(
            model=EVAL_MODEL,
            max_tokens=MAX_TOKENS,
            system=BIBLE_LOOKUP_PROMPT.format(
                bible_name=bible_name,
                target_language=target_language,
            ),
            messages=[{
                "role": "user",
                "content": f"Provide the exact {target_language} text from the {bible_name} for these verses:\n\n{refs_text}"
            }],
        )
    except Exception as api_err:
        # The Anthropic SDK raises anthropic.BadRequestError,
        # anthropic.APIError, anthropic.RateLimitError, etc. We catch them
        # all with Exception on purpose — any failure here must NOT crash
        # a multi-chunk translation run. Record it and let the translator
        # produce its own rendering inline (flagged for human review).
        err_type = type(api_err).__name__
        err_msg = str(api_err)
        print(f"  WARNING: Bible lookup failed ({err_type}): {err_msg[:200]}")
        print(f"  -> Falling back: translator will render these verses "
              f"inline using its own knowledge, and the references have "
              f"been added to the human-review queue.")
        _record_bible_lookup_failure(
            chunk_id=chunk_id,
            references=references,
            target_language=target_language,
            error_type=err_type,
            error_message=err_msg,
        )
        return {}

    response_text = response.content[0].text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

    try:
        result = json.loads(response_text)
        verses = {}
        for v in result.get("verses", []):
            ref = v.get("reference", "")
            verses[ref] = {
                "text": v.get("text", ""),
                "confidence": v.get("confidence", "unknown"),
                "bible": bible_name,
            }
        return verses
    except json.JSONDecodeError:
        print(f"  WARNING: Failed to parse Bible lookup response — "
              f"logging for human review and continuing")
        _record_bible_lookup_failure(
            chunk_id=chunk_id,
            references=references,
            target_language=target_language,
            error_type="JSONDecodeError",
            error_message="Response from Bible lookup was not valid JSON",
        )
        return {}


# =========================================================================
# MAIN FUNCTION: Process quotes for a chunk
# =========================================================================

def process_quotes_for_chunk(chunk_text, target_language, source_language=None,
                              use_ai_detection=True, chunk_id=None):
    """
    The main function for quote handling. For a given chunk:
    1. Detect Bible references (regex + optionally AI)
    2. Detect other quotes (AI)
    3. Look up real Bible translations for the target language
    4. Return a mapping the translator can use

    Parameters:
        chunk_text (str): The source text
        target_language (str): Target language for Bible lookups
        source_language (str): Source language override
        use_ai_detection (bool): Whether to use AI for quote detection
                                 (more thorough but costs API calls)

    Returns:
        dict: {
            "bible_verses": {reference: {text, confidence, bible}},
            "other_quotes": [list of non-Bible quotes found],
            "quote_instructions": str  # formatted text for the translator prompt
        }
    """
    lang = source_language or SOURCE_LANGUAGE

    # Step 1: Detect Bible references with regex (fast, free)
    print(f"  Detecting Bible references...")
    regex_refs = detect_bible_references(chunk_text)
    print(f"  Regex found {len(regex_refs)} Bible references")

    # Step 2: Optionally use AI for more thorough detection
    all_quotes = []
    if use_ai_detection:
        print(f"  Running AI quote detection...")
        all_quotes = detect_all_quotes(chunk_text, lang)
        print(f"  AI found {len(all_quotes)} total quotes")

    # Combine Bible references from both methods
    bible_refs = set()
    for ref in regex_refs:
        bible_refs.add(ref["standardized"])

    for quote in all_quotes:
        if quote.get("type") == "bible" and quote.get("standardized_ref"):
            bible_refs.add(quote["standardized_ref"])

    bible_refs = sorted(bible_refs)
    print(f"  Total unique Bible references: {len(bible_refs)}")

    # Step 3: Look up real Bible translations
    bible_verses = {}
    if bible_refs:
        print(f"  Looking up {len(bible_refs)} verses in {target_language} Bible...")
        bible_verses = lookup_bible_verses(bible_refs, target_language, chunk_id=chunk_id)
        confident = sum(1 for v in bible_verses.values() if v["confidence"] == "certain")
        print(f"  Found {len(bible_verses)} verses ({confident} high-confidence)")

    # Step 4: Collect non-Bible quotes
    other_quotes = [q for q in all_quotes if q.get("type") != "bible"]

    # Step 5: Build instruction text for the translator
    instructions_parts = []

    if bible_verses:
        instructions_parts.append("BIBLE QUOTES — Use these EXACT translations from the published Bible:")
        for ref, data in bible_verses.items():
            confidence_note = ""
            if data["confidence"] == "uncertain":
                confidence_note = " [APPROXIMATE — verify if possible]"
            instructions_parts.append(
                f"  {ref} ({data['bible']}): {data['text']}{confidence_note}"
            )

    if other_quotes:
        instructions_parts.append("\nOTHER QUOTES — Translate these with extra care for accuracy:")
        for q in other_quotes:
            instructions_parts.append(
                f"  [{q.get('type', 'unknown')}] {q.get('reference', 'unknown source')}: "
                f"\"{q.get('quoted_text', '')[:100]}...\""
            )

    quote_instructions = "\n".join(instructions_parts) if instructions_parts else ""

    return {
        "bible_verses": bible_verses,
        "other_quotes": other_quotes,
        "quote_instructions": quote_instructions,
    }
