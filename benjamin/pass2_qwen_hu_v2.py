"""
pass2_qwen_hu.py
Pass 2 — Semantic enrichment and correction of MadLAD Hungarian base translation
for Walther's *The Proper Distinction Between Law and Gospel*.

This pass receives:
  - The German source text (authoritative reference)
  - The MadLAD base translation (structural skeleton to correct)
  - Per-chunk annotation metadata (summary, tone, argument structure, rhetorical devices)
  - Per-chunk glossary terms extracted from annotations (term → Hungarian + notes)
  - The tail of the previous chunk's German source (for boundary continuations)
  - Lecture and thesis context from chunk metadata

This pass corrects:
  - Theological term errors against the glossary
  - Semantic inversions and polarity errors
  - Citation errors (wrong verse numbers, spurious inline verse numbers)
  - Dropped content (sentences, clauses, rhetorical questions)
  - Appositive and structural relationship losses
  - Grammatical errors introduced by MadLAD
  - Chunk boundary continuation fragments

This pass does NOT handle:
  - Register normalization (formal Sie vs informal — handled in Pass 3)
  - Stylistic refinement (handled in Pass 3)
  - Károli alignment beyond direct biblical quotations (handled in Pass 3)

Input:
  - chunks.json              : JSON array of {chunk_id, lecture, thesis, text}
  - annotated/<id>.json      : per-chunk annotation metadata
  - hu_madlad/pass1/<id>.txt : MadLAD base translation

Output:
  - hu_qwen/pass2/<id>.txt   : semantically corrected Hungarian

Usage:
  python pass2_qwen_hu.py [start] [end] [max_workers]
  python pass2_qwen_hu.py 1 50 4
"""

import sys
import os
import json
import re
import time
import spacy
from rapidfuzz.distance import Levenshtein
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

nlp = spacy.load("de_core_news_sm")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR      = os.path.expanduser("~/ge_rig")
CHUNKS_FILE   = os.path.join(BASE_DIR, "output", "chunks.json")
ANNOTATED_DIR = os.path.join(BASE_DIR, "output", "annotated")
MADLAD_DIR    = os.path.join(BASE_DIR, "output", "hu_madlad", "pass1")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output", "hu_qwen")
GLOSSARY_PATH = os.path.join(BASE_DIR, "output", "glossaries", "glossary_hu.json")

MODEL  = "Qwen/Qwen3.5-27B-FP8"  # update to your served model name
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-not-needed")

# How many characters of the previous chunk's German to include for boundary context
PREV_CHUNK_TAIL_CHARS = 250

_GLOSSARY = []
if os.path.exists(GLOSSARY_PATH):
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        _GLOSSARY = json.load(f)

_FUNCTION_POS = {"ADP", "DET", "PRON", "CCONJ", "SCONJ", "PART", "PUNCT", "SPACE", "AUX"}

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """\
You are a theological editor working on a critical Hungarian edition of C. F. W. \
Walther's *The Proper Distinction Between Law and Gospel* \
(19th-century Lutheran German, originally delivered as evening lectures 1884–1885).

You will receive:
1. A GERMAN SOURCE TEXT — this is the authoritative reference. Every word in the \
German must be accounted for in your output.
2. A MADLAD DRAFT — a Hungarian base translation produced by a machine translation \
model. It provides a structural skeleton but contains errors you must correct.
3. CHUNK METADATA — summary, tone, argument structure, and rhetorical devices for \
this passage.
4. A GLOSSARY — theological and other vocabulary terms with their correct Hungarian \
renderings and translation notes. These renderings are binding. You must use them \
exactly as given.

CONTEXT-DEPENDENT TERMS:
Some glossary entries are marked [USE CONTEXT]. For these, you must pick the choice \
that best matches the sentence meaning.

5. PREVIOUS CHUNK TAIL — the last portion of the preceding German passage. Use this \
to resolve sentences that begin mid-argument or mid-quotation.

YOUR TASK is to produce a corrected Hungarian translation that:
— Is semantically faithful to the German source in every sentence
— Uses glossary terms exactly as provided, with correct Hungarian inflection
— Preserves all of Walther's rhetorical structures, antitheses, and argument steps
— Preserves all Latin, Greek, and Hebrew text exactly as it appears in the German
— Uses Károli Bible phrasing for all biblical quotations
— Preserves citation format exactly as it appears in the German — do not add, \
remove, or modify verse numbers from memory
— Resolves any chunk boundary fragments by reference to the previous chunk tail
— Corrects all grammatical errors in the MadLAD draft

CRITICAL CORRECTION RULES — apply these before anything else:

1. THEOLOGICAL TERM POLARITY
   Always verify the logical polarity of statements about spiritual states, \
   grace, Law, and Gospel against the German. The following pairs are \
   frequently confused by the MadLAD model:
   — újjászületett (regenerate) vs. újjá nem született (unregenerate)
   — feltétel nélküli (unconditional) vs. feltételes (conditional)
   — törvény (Law) vs. evangélium (Gospel) as subjects of predications
   If the German says something is WITHOUT condition (ohne Bedingung, ohne alle \
   Bedingung) the Hungarian must say feltétel nélkül — never "nem feltétel nélküli".
   If the MadLAD draft inverts any of these, correct it immediately.

2. RECHT THEILEN — THE STRUCTURAL THESIS MARKER
   The phrase "das Wort Gottes recht theilen" and its negation \
   "nicht recht getheilt" are Walther's central thesis construction throughout \
   the entire book, drawn from 2 Tim 2:15 (Luther: "das Wort der Wahrheit recht \
   theile"). It must always be rendered as:
   — "Isten igéjét helyesen felosztani" or "helyesen megkülönböztetni"
   — NEVER as "értelmezni" (to interpret), "magyarázni" (to explain), or any \
   other verb that loses the sense of division/distinction between Law and Gospel.
   The numbered thesis markers (tizenkilencedszer, huszadszor, etc.) must be \
   preserved exactly.

3. CITATION FAITHFULNESS
   Copy citation references EXACTLY from the German source.
   — Do not add verse numbers that are not in the German
   — Do not correct or supply verse numbers from memory — MadLAD frequently \
   produces wrong verse numbers and you must use the German as ground truth
   — Do not expand abbreviated book names (Jer. stays Jer., not Jeremiás)
   — Do not insert inline verse numbers before Károli quotations unless the \
   German has them
   — Spurious Adventist source tags like {DA 102.3} or {LDE 49.1} must be \
   removed — these are MadLAD hallucinations with no basis in the German

4. APPOSITIVE THEOLOGICAL IDENTIFICATIONS
   When Walther uses apposition to identify two theological concepts \
   (e.g. "der Geist, das Evangelium" — the Spirit, that is, the Gospel), \
   preserve the identifying relationship using "vagyis" or an em-dash:
   "a Lélek — vagyis az evangélium". Do not flatten these to lists or \
   predicate constructions.

5. CHUNK BOUNDARY CONTINUATIONS
   If the chunk opens with a very short sentence or a pronoun without a clear \
   antecedent (e.g. "Ez öl." / "Das tödtet."), check the PREVIOUS CHUNK TAIL \
   to establish the subject. Restore any rhetorical questions or transitional \
   clauses that the MadLAD draft has dropped at the boundary. Subject-dropped \
   German sentences (finite verb without explicit subject) must be given an \
   explicit Hungarian subject resolved from context.

6. GLOSSARY TERMS ARE BINDING
   Every term in the provided GLOSSARY must appear in the output with the \
   exact Hungarian rendering given, inflected correctly for its grammatical \
   context. If the MadLAD draft uses a different rendering for a glossary term, \
   replace it. Translation notes in the glossary explain why a specific \
   rendering is required — read them carefully.

7. DROPPED CONTENT
   Compare each German sentence against the MadLAD draft. If a sentence, \
   clause, or rhetorical question present in the German has no counterpart in \
   the draft, restore it by translating directly from the German. Pay particular \
   attention to:
   — Short rhetorical questions following longer statements
   — Transitional sentences opening new argument sections
   — The second half of antithetical constructions (Gesetz... aber Evangelium...)
   — Parenthetical qualifiers (im besten Fall, freilich, zwar)

8. GRAMMATICAL CORRECTION
   Correct any grammatical errors in the MadLAD draft including:
   — Wrong case endings on theological compounds
   — Mismatched number agreement
   — Incorrect verbal aspect or tense
   — Confused quotation attribution (ensure quoted speech is attributed to the \
   correct speaker)
   — Malformed compounds (e.g. "vasban-acélban" for "Stahl und Eisen" — \
   use the natural Hungarian equivalent or the Károli idiom)

9. GERMAN VOCABULARY
   All German words in the MadLAD draft that have not been translated must be
   rendered into Hungarian. Do not pass through German vocabulary.
   Common cases include adjectives like "starrblind" (elvakult/megvakult) and
   archaic German forms that MadLAD fails to translate.

   The abbreviation ꝛc. (a medieval abbreviation for "et cetera") should be
   rendered as "stb." in Hungarian

10. LATIN TECHNICAL TERMS
    Walther uses certain Latin scholastic terms as fixed technical vocabulary.
    These must be preserved exactly as they appear in the German source.
    If the glossary provides a Hungarian rendering for a Latin term, place
    it in parentheses immediately after the Latin, do not replace the Latin
    with the Hungarian. For example:
    — subjectum operationis (a cselekvés alanya)
    — finis cui (a végső cél)
    — formali causa (formai ok)
    — in causa formali (a formai ok tekintetében)
    Do not translate the Latin into Hungarian without also preserving it.
    The Latin must always appear first, with the Hungarian gloss optional
    and in parentheses.

DO NOT include translator's notes, commentary, explanations, or any text \
other than the corrected Hungarian translation.
Do not translate Latin, Greek, or Hebrew — preserve them exactly as in the German.
Output language: Hungarian. Use Károli Bible register throughout.

=== FEW-SHOT CORRECTION EXAMPLES ===

--- EXAMPLE 1: Theological term precision and polarity ---

GERMAN:
Das Wort Gottes wird neunzehntens nicht recht getheilt, wenn man die \
Unwiedergebornen durch die Forderungen oder Drohungen oder Verheißungen \
des Gesetzes zur Ablegung von Sünden und zu guten Werken zu bewegen, und \
also fromm zu machen, die Wiedergebornen aber, anstatt sie evangelisch zu \
ermahnen, durch gesetzliches Gebieten zum Guten zu nöthigen sucht.

MADLAD DRAFT:
Isten szavát nem helyesen értelmezik, ha az újjászületetteket evangéliumi \
buzdítás helyett a törvény követeléseivel, fenyegetéseivel vagy ígéreteivel \
arra kényszerítik, hogy hagyják el bűneiket és tegyenek jó cselekedeteket.

CORRECT OUTPUT:
Tizenkilencedszer Isten igéjét nem osztják fel helyesen, ha az újjá nem \
születetteket a törvény követeléseivel, fenyegetéseivel vagy ígéreteivel \
próbálják bűneik elhagyására és jó cselekedetekre indítani, és ezáltal \
kegyessé tenni; a már újjászületetteknél pedig ahelyett, hogy evangéliumi \
intéssel buzdítanák őket, törvényes parancsolgatással kívánják őket a jóra \
kényszeríteni.

ERRORS CORRECTED:
(a) "recht theilen" rendered as "helyesen felosztani/megkülönböztetni," not \
"értelmezni." This is Walther's structural thesis marker — it must always \
reference the Law/Gospel distinction, not interpretation.
(b) "Unwiedergebornen" (unregenerate) corrected from "újjászületetteket" \
(regenerate) — MadLAD inverted the spiritual state entirely.
(c) The second clause about the regenerate ("die Wiedergebornen aber...") was \
dropped by MadLAD and has been restored.
(d) Thesis ordinal "Tizenkilencedszer" restored.

--- EXAMPLE 2: Citation faithfulness ---

GERMAN:
Jer. 31, 31—34.: „Siehe, kommt die Zeit, spricht der Herr, da will ich mit \
dem Hause Israel und mit dem Hause Juda einen neuen Bund machen."

MADLAD DRAFT:
Jeremiás próféta könyve 31:29 Ímé eljőnek a napok, azt mondja az Úr; és új \
szövetséget kötök az Izráel házával és Júda házával.

CORRECT OUTPUT:
Jer. 31, 31—34.: „Ímé eljőnek a napok, azt mondja az Úr; és új szövetséget \
kötök az Izráel házával és Júda házával."

ERRORS CORRECTED:
(a) Verse number corrected from 31:29 to 31:31 per the German source.
(b) Citation format restored to abbreviated German form "Jer. 31, 31—34." — \
do not expand to full Hungarian book name.
(c) Károli Bible phrasing preserved — this is correct and should be kept \
whenever MadLAD produces recognizable Károli text for a biblical quotation.

--- EXAMPLE 3: Appositive theological identification ---

GERMAN:
Das ist kein Wunder, wenn man dann nichts wirkt; denn das Gesetz tödtet, \
aber der Geist, das Evangelium, macht lebendig.

MADLAD DRAFT:
Mert a törvény halált okoz, de a Lélek az evangélium életet ad.

CORRECT OUTPUT:
Nem csoda hát, ha semmi hatása nincs; mert a törvény öl, de a Lélek — \
vagyis az evangélium — életet ad.

ERRORS CORRECTED:
(a) Opening clause "Das ist kein Wunder, wenn man dann nichts wirkt" dropped \
by MadLAD — restored.
(b) "der Geist, das Evangelium" is an appositive identifying the Spirit with \
the Gospel. Rendered as "a Lélek — vagyis az evangélium" to preserve the \
theological identification. The MadLAD rendering "a Lélek az evangélium" \
reads as either a predicate ("the Spirit is the Gospel") or a list, losing \
the appositive sense.

--- EXAMPLE 4: Semantic inversion of Gospel promise ---

GERMAN:
Das verheißt uns Gottes Gnade und Seligkeit ohne alle Bedingung. \
ist eine Verheißung freier Gnade. begehrt nichts, als: „Nimm an, dann hast du es."

MADLAD DRAFT:
A kegyelem és a szeretet nem feltétel nélküli, hanem csak úgy adható: \
„Fogadd el, akkor megkapod."

CORRECT OUTPUT:
Ez Isten kegyelmét és üdvösségét ígéri minden feltétel nélkül. \
Ez a szabad kegyelem ígérete. \
Nem kíván mást, mint: „Fogadj el, és megkapod."

ERRORS CORRECTED:
(a) "nem feltétel nélküli" (not unconditional) inverts the German "ohne alle \
Bedingung" (without any condition). The Gospel promise is unconditional — \
"feltétel nélkül" with no negation.
(b) "Seligkeit" (salvation/blessedness) dropped from MadLAD draft — restored \
as "üdvösség."
(c) "szeretet" (love) has no basis in the German — removed.
(d) Three separate German sentences restored as three separate Hungarian \
sentences. MadLAD collapsed them into one.

--- EXAMPLE 5: Chunk boundary continuation ---

PREVIOUS CHUNK TAIL (German):
...denn das Gesetz tödtet, aber der Geist, das Evangelium, macht lebendig.

CURRENT CHUNK GERMAN OPENING:
Das tödtet. Wenn aber tödtet, wie kann uns dann fromm machen? \
kann uns wohl dahin bringen, daß wir diese und jene äußerlichen Laster lassen, \
aber das Herz kann nicht ändern.

MADLAD DRAFT OPENING:
Ez öl.
Az erkölcsi törvény arra vezethet, hogy elhagyjuk ezeket és azokat a külső \
bűnöket, de nem képes megváltoztatni a szívet.

CORRECT OUTPUT:
Ez öl. De ha öl, hogyan tehetne bennünket kegyessé? \
Elvezet ugyan ahhoz, hogy elhagyjuk ezt vagy azt a külső bűnt, \
de a szívet nem tudja megváltoztatni.

ERRORS CORRECTED:
(a) "Wenn aber tödtet, wie kann uns dann fromm machen?" — the rhetorical \
question was dropped by MadLAD entirely. Restored: "De ha öl, hogyan \
tehetne bennünket kegyessé?"
(b) The subject of "tödtet" is "das Gesetz" carried forward from the previous \
chunk — resolved explicitly in context.
(c) "Das Herz kann nicht ändern" — "das Herz" is the object, not the subject. \
Corrected word order in Hungarian: "a szívet nem tudja megváltoztatni."

--- EXAMPLE 6: Lexical ambiguity resolved by theological context ---

GERMAN:
Der Teufel hat das menschliche Geschlecht greulich entstellt und in tiefes
Elend gestürzt. Das hat Christus dann gerochen.

MADLAD DRAFT:
Az ördög szörnyen eltorzította az emberi nemet, és mély nyomorba taszította.
Ez volt Krisztus illata.

CORRECT OUTPUT:
Az ördög szörnyen eltorzította az emberi nemet, és mély nyomorba taszította.
Ezt Krisztus megbosszulta.

NOTE: "gerochen" is the past participle of both "riechen" (to smell) and
"rächen" (to avenge) in 19th-century German. MadLAD has chosen the "smell"
reading, producing "Ez volt Krisztus illata" (This was Christ's smell) which
is theologically incoherent. The correct reading is "rächen" (to avenge) —
Christ avenged what the devil did to humanity. The theological context makes
this unambiguous: the subject is Christ, the object is the devil's destruction
of the human race, and the entire passage concerns Christ's redemptive
conquest.

"""

# =============================================================================
# GLOSSARY AND ANNOTATION (reused from translate_max_small.py)
# =============================================================================

def _content_lemmas(phrase: str, nlp) -> frozenset:
    """
    Extract lemmatized content words from a phrase string.
    Excludes function words, punctuation, and single-character tokens.
    """
    doc = nlp(phrase)
    return frozenset(
        tok.lemma_.lower()
        for tok in doc
        if tok.pos_ not in _FUNCTION_POS
        and len(tok.lemma_) > 1
        and not tok.is_punct
    )


def _build_glossary_index(glossary: list, nlp, lang_field: str = "hungarian"):
    """
    Split glossary into single-token and phrase entries.
    Pre-compute content lemmas for phrase entries.
    Returns (single_index, phrase_entries).
    
    single_index: dict of headword_lower -> entry
    phrase_entries: list of (frozenset_of_content_lemmas, min_window, entry)
    """
    single_index = {}
    phrase_entries = []

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
            lemmas = _content_lemmas(term, nlp)
            if lemmas:
                # Window size: number of words in phrase + 3 for flexibility
                window = len(words) + 3
                phrase_entries.append((lemmas, window, entry))

    return single_index, phrase_entries


# Module-level cache so we only build the index once per process
_GLOSSARY_INDEX_CACHE = None
_GLOSSARY_INDEX_LOCK = None

def _get_glossary_index(glossary: list, nlp, lang_field: str = "hungarian"):
    global _GLOSSARY_INDEX_CACHE, _GLOSSARY_INDEX_LOCK
    import threading
    if _GLOSSARY_INDEX_LOCK is None:
        _GLOSSARY_INDEX_LOCK = threading.Lock()
    with _GLOSSARY_INDEX_LOCK:
        if _GLOSSARY_INDEX_CACHE is None:
            _GLOSSARY_INDEX_CACHE = _build_glossary_index(glossary, nlp, lang_field)
    return _GLOSSARY_INDEX_CACHE


def get_relevant_glossary(
    german_text: str,
    glossary: list,
    max_distance: int = 3,
    max_per_source_token: int = 2,
    lang_field: str = "hungarian",
) -> str:
    """
    Fuzzy glossary lookup combining:
    1. Levenshtein single-token matching (lemmatized source tokens)
    2. Bag-of-lemmas window matching for phrase entries
    """
    single_index, phrase_entries = _get_glossary_index(glossary, nlp, lang_field)

    # Lemmatize source text
    doc = nlp(german_text)
    source_tokens = [
        tok for tok in doc
        if not tok.is_punct and not tok.is_space
    ]
    source_lemmas_list = [tok.lemma_.lower() for tok in source_tokens]
    source_lemmas_set  = set(source_lemmas_list)

    results     = []
    seen_headwords = set()

    # ── Phase 1: phrase matching (bag-of-lemmas within window) ──
    for content_lemmas, window, entry in phrase_entries:
        term = entry.get("term", "").lower()
        if term in seen_headwords:
            continue
        if not content_lemmas.issubset(source_lemmas_set):
            # Fast rejection: not all content lemmas present anywhere in text
            continue
        # Sliding window check
        matched = False
        for i in range(len(source_lemmas_list)):
            window_slice = set(source_lemmas_list[i : i + window])
            if content_lemmas.issubset(window_slice):
                matched = True
                break
        if matched:
            seen_headwords.add(term)
            results.append(entry)

    # ── Phase 2: single-token Levenshtein matching ──
    seen_source_lemmas = set()
    headwords = list(single_index.keys())

    for lemma in source_lemmas_list:
        if lemma in seen_source_lemmas:
            continue
        seen_source_lemmas.add(lemma)

        n     = len(lemma)
        max_d = 1 if n <= 4 else (2 if n <= 7 else 3)

        candidates = []
        for hw in headwords:
            if hw in seen_headwords:
                continue
            dist = Levenshtein.distance(lemma, hw, score_cutoff=max_d)
            if dist <= max_d:
                candidates.append((dist, hw))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])
        for _, hw in candidates[:max_per_source_token]:
            if hw not in seen_headwords:
                seen_headwords.add(hw)
                results.append(single_index[hw])

    if not results:
        return "No glossary terms available."

    # ── Format: term → hungarian only, no notes ──
    lines = []
    for entry in results:
        term = entry.get("term", "")
        hu   = entry.get(lang_field, "")
        if term and hu:
            lines.append(f"{term} → {hu}")

    return "\n".join(lines)

def load_annotation(data: dict) -> str:
    if not data:
        return ""

    parts = []

    if data.get("summary"):
        parts.append(f"Summary: {data['summary']}")
    if data.get("tone"):
        parts.append(f"Tone: {data['tone']}")

    arg = data.get("argument_structure", [])
    if arg:
        steps = "\n".join(
            f"  {s.get('step', i+1)}. {s.get('description', '')}"
            for i, s in enumerate(arg)
            if isinstance(s, dict)
        )
        parts.append(f"Argument structure:\n{steps}")

    devices = data.get("rhetorical_devices", [])
    if devices:
        lines = []
        for d in devices:
            if not isinstance(d, dict):
                continue
            phrase     = d.get("german_phrase", "")
            dtype      = d.get("device_type", "")
            explain    = d.get("english_explanation", "")
            line = f"  [{dtype}] \"{phrase}\": {explain}"
            lines.append(line)
        if lines:
            parts.append("Rhetorical devices:\n" + "\n".join(lines))

    return "\n\n".join(parts)


# =============================================================================
# MODEL CALL
# =============================================================================

def _call(messages: list, timeout: int = 600) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,          # lower than translate_max_small — correction not generation
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=timeout,
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# PASS 2 LOGIC
# =============================================================================

def enrich_chunk(chunk: dict, prev_chunk: dict | None) -> tuple:
    """
    Run Pass 2 semantic enrichment on a single chunk.
    Returns (chunk_id, elapsed_seconds_or_None, error_string_or_None).
    """
    chunk_id   = str(chunk["chunk_id"])
    chunk_text = chunk["text"].strip()
    lecture    = chunk.get("lecture", "")
    thesis     = chunk.get("thesis", "")

    out_path = os.path.join(OUTPUT_DIR, "pass2", f"{chunk_id}.txt")
    if os.path.exists(out_path):
        return chunk_id, None, None

    # Load MadLAD base translation
    madlad_path = os.path.join(MADLAD_DIR, f"{chunk_id}.txt")
    if not os.path.exists(madlad_path):
        return chunk_id, None, f"MadLAD output missing for {chunk_id}"

    with open(madlad_path, "r", encoding="utf-8") as f:
        madlad_draft = f.read().strip()

    # Load annotation data
    annotation_data = {}
    annotated_path  = os.path.join(ANNOTATED_DIR, f"{chunk_id}.json")
    if os.path.exists(annotated_path):
        with open(annotated_path, "r", encoding="utf-8") as f:
            annotation_data = json.load(f)

    annotation     = load_annotation(annotation_data)
    relevant_terms = get_relevant_glossary(chunk_text, _GLOSSARY)

    # Build context header
    context_header = ""
    if lecture:
        context_header += f"Lecture: {lecture}\n"
    if thesis:
        context_header += f"Thesis: {thesis}\n"

    # Previous chunk tail for boundary context
    prev_tail = ""
    if prev_chunk:
        prev_text = prev_chunk.get("text", "").strip()
        if prev_text:
            prev_tail = prev_text[-PREV_CHUNK_TAIL_CHARS:]

    # Assemble user content
    user_content = ""

    if context_header:
        user_content += f"{context_header}\n"

    if prev_tail:
        user_content += f"PREVIOUS CHUNK TAIL (German — for boundary context only):\n{prev_tail}\n\n"

    user_content += f"SOURCE TEXT (German):\n{chunk_text}\n\n"

    if annotation:
        user_content += f"CHUNK METADATA:\n{annotation}\n\n"

    user_content += f"GLOSSARY:\n{relevant_terms}\n\n"
    user_content += f"MADLAD DRAFT (Hungarian — correct this):\n{madlad_draft}\n\n"
    user_content += "Produce the corrected Hungarian translation now."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    try:
        t0     = time.time()
        result = _call(messages, timeout=600)
        elapsed = time.time() - t0
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)
            # DEBUG: append matched glossary terms after translation output
            # Remove this block once glossary lookup is validated
            f.write("\n\n")
            f.write("=" * 60 + "\n")
            f.write("DEBUG — GLOSSARY TERMS INJECTED FOR THIS CHUNK\n")
            f.write("=" * 60 + "\n")
            if relevant_terms and relevant_terms != "No glossary terms available.":
                f.write(relevant_terms)
            else:
                f.write("(none)\n")
        return chunk_id, elapsed, None
    except Exception as e:
        return chunk_id, None, f"Pass 2 error: {e}"


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run(start_idx: int, end_idx: int, max_workers: int):
    os.makedirs(os.path.join(OUTPUT_DIR, "pass2"), exist_ok=True)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    all_chunks.sort(key=lambda c: int(str(c["chunk_id"])))
    target_chunks = all_chunks[start_idx - 1 : end_idx]

    # Build lookup for previous chunk access
    full_sorted = sorted(all_chunks, key=lambda c: int(str(c["chunk_id"])))
    chunk_index = {str(c["chunk_id"]): i for i, c in enumerate(full_sorted)}

    print(f"Pass 2 enrichment | language: Hungarian | chunks {start_idx}–{end_idx} "
          f"({len(target_chunks)} chunks) | workers: {max_workers}")
    print(f"Model: {MODEL}")
    print(f"Output: {OUTPUT_DIR}\n")

    pending, skipped = [], 0
    for chunk in target_chunks:
        cid = str(chunk["chunk_id"])
        if os.path.exists(os.path.join(OUTPUT_DIR, "pass2", f"{cid}.txt")):
            skipped += 1
        else:
            pending.append(chunk)

    if skipped:
        print(f"Skipping {skipped} already-completed chunks.")
    print(f"Processing {len(pending)} chunks.\n")

    def task(chunk):
        cid = str(chunk["chunk_id"])
        idx = chunk_index.get(cid, 0)
        prev_chunk = full_sorted[idx - 1] if idx > 0 else None
        return enrich_chunk(chunk, prev_chunk)

    times  = []
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, c): c["chunk_id"] for c in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Pass 2"):
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
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    run(start, end, workers)
