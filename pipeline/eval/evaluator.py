"""
evaluator.py - The unified evaluation engine for the WELS Translation Pipeline

This evaluator works with ANY translation source — our pipeline, Benjamin's
pipeline, or any external translator. Feed it a source text and a translation,
and it produces a rigorous quality assessment.

THREE EVALUATION METHODS (used together for maximum reliability):
  1. Rubric Scoring - AI scores the translation on 4 weighted dimensions.
     Run TWICE with slightly different prompts to check consistency.
     If the two passes disagree by more than 1 point on any dimension,
     a third tiebreaker pass is triggered.

  2. Back-Translation - Translate the output BACK to the source language,
     then compare with the original. Catches meaning loss that rubric
     scoring might miss.

  3. Consistency Check - Compares the two rubric passes and flags any
     dimension where scores diverged, so you know which scores to trust
     and which to investigate.

WHY THIS MATTERS:
  A single AI evaluation can be inconsistent — ask the same question twice
  and you might get different scores. By running two passes and comparing,
  we catch that inconsistency and resolve it. The final score is the one
  you can trust.
"""

import json
import re
from anthropic import Anthropic

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    EVAL_MODEL,
    BACK_TRANSLATE_MODEL,
    MAX_TOKENS,
    EVAL_WEIGHTS,
    SOURCE_LANGUAGE,
)


def get_client():
    """Creates a connection to the Anthropic API."""
    return Anthropic()


def _parse_json_response(text):
    """
    Robust JSON parser that handles common AI response quirks:
    - Markdown code blocks wrapping
    - Truncated responses (try to close brackets)
    - Control characters that break json.loads
    - Braces inside quoted strings (won't confuse depth tracking)
    - Very long responses that get cut off mid-string
    """
    # Strip markdown code blocks
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Remove any leading "json" label
    if text.startswith("json"):
        text = text[4:].strip()

    # Try standard parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove control characters that break JSON (but keep newlines in strings)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object with STRING-AWARE brace matching
    # This correctly handles braces inside quoted strings like "{'term'}"
    match = re.search(r'\{', cleaned)
    if match:
        json_start = match.start()
        depth = 0
        in_string = False
        escape_next = False
        last_close = -1

        for i, c in enumerate(cleaned[json_start:], json_start):
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue  # Skip everything inside strings
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                last_close = i
                if depth == 0:
                    break

        if last_close > json_start:
            candidate = cleaned[json_start:last_close + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Try the simple first-{ to last-} approach as fallback
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Last resort: try closing truncated JSON
    # Find the start of the JSON and work with everything after it
    json_start_idx = cleaned.find('{')
    if json_start_idx != -1:
        truncated = cleaned[json_start_idx:]

        # Count structural braces/brackets (outside of strings)
        open_braces = 0
        open_brackets = 0
        in_str = False
        esc = False
        for c in truncated:
            if esc:
                esc = False
                continue
            if c == '\\' and in_str:
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == '{': open_braces += 1
            elif c == '}': open_braces -= 1
            elif c == '[': open_brackets += 1
            elif c == ']': open_brackets -= 1

        if open_braces > 0 or open_brackets > 0:
            patched = truncated.rstrip().rstrip(',')
            # Close any open strings
            if patched.count('"') % 2 == 1:
                patched += '"'
            patched += ']' * max(0, open_brackets) + '}' * max(0, open_braces)
            try:
                return json.loads(patched)
            except json.JSONDecodeError:
                pass

    return {"error": "Failed to parse JSON after all attempts", "raw_response": text[:2000]}


def _extract_scores_from_raw(text):
    """
    Pulls numeric scores from the raw AI response using regex.
    This gives us a ground truth to verify against after recovery.
    Returns dict like {"doctrinal_accuracy": 4, "terminology_consistency": 3, ...}
    or empty dict if we can't find scores.
    """
    scores = {}
    dimensions = [
        "doctrinal_accuracy",
        "terminology_consistency",
        "clarity",
        "naturalness",
    ]
    for dim in dimensions:
        # Look for patterns like: "score": 4  or  "score": 5
        # near the dimension name in the raw text
        # Strategy: find the dimension name, then find the nearest "score": N
        dim_pos = text.find(dim)
        if dim_pos == -1:
            # Try with spaces instead of underscores
            dim_pos = text.find(dim.replace("_", " "))
        if dim_pos == -1:
            continue

        # Search for "score": N within 500 chars after the dimension name
        chunk = text[dim_pos:dim_pos + 500]
        score_match = re.search(r'"score"\s*:\s*(\d+)', chunk)
        if score_match:
            scores[dim] = int(score_match.group(1))

    return scores


def _verify_recovered_scores(parsed, raw_scores, original_text):
    """
    Checks that the recovered JSON has the same scores as what we
    extracted from the raw text. If any score was changed during
    recovery, we force it back to the original value.
    """
    changed = False
    for dim, raw_score in raw_scores.items():
        recovered_score = parsed.get(dim, {}).get("score")
        if recovered_score is not None and recovered_score != raw_score:
            print(f"    [SCORE GUARD] Recovery changed {dim} from {raw_score} to "
                  f"{recovered_score} — forcing back to {raw_score}")
            parsed[dim]["score"] = raw_score
            changed = True

    if not changed and raw_scores:
        print(f"    [SCORE GUARD] Verified — all scores match original")

    return parsed


# =========================================================================
# RUBRIC SYSTEM PROMPTS — Two variants for consistency checking
# =========================================================================

def _build_rubric_prompt(variant="A"):
    """
    Builds the rubric evaluation prompt. Two variants exist so we can
    run the evaluation twice with slightly different framing and check
    if the scores are consistent. If they agree, we have high confidence.
    If they disagree, we investigate.
    """
    pipeline_context = f"""PIPELINE CONTEXT: You are Step 5 in a translation pipeline for the
Wisconsin Evangelical Lutheran Synod (WELS). Before you, an AI translator (Step 4) produced
a translation using an approved glossary and full document context. Your evaluation scores
will be used to decide whether the translation is good enough (5/5 on doctrine and
terminology) or needs to go back for targeted fixes (Step 6). Your scores directly
control that decision — do NOT inflate or deflate them. Score exactly what you see.

YOUR ROLE: You are ONLY an evaluator. You do NOT translate, improve, or suggest
alternative translations. You assess what is in front of you and report honestly.
Do NOT let the quality of one dimension influence your score on another dimension.
Score each independently."""

    if variant == "A":
        intro = f"""{pipeline_context}

You are a senior theological translation reviewer with expertise in
Lutheran confessional documents (Book of Concord, WELS texts). You have deep knowledge
of both the source language ({SOURCE_LANGUAGE}) and the target language.

Evaluate this translation with the rigor of a seminary professor reviewing a
confessional text for official church use. Every doctrinal nuance matters."""
    else:
        intro = f"""{pipeline_context}

You are an expert linguist and theologian specializing in translating
Reformation-era confessional documents. You understand that these are binding doctrinal
texts where even subtle shifts in meaning can create false teaching.

Evaluate this translation as if it will be used by pastors and congregations
who depend on its accuracy for their faith and practice."""

    return f"""{intro}

CRITICAL CONTEXT: These are doctrinal texts where theological precision is paramount.
A translation that sounds beautiful but subtly shifts a doctrinal meaning is WORSE
than one that sounds awkward but preserves the theology perfectly. The Lutheran
confessions are precisely worded — every phrase carries doctrinal weight.

SCORING DIMENSIONS (score each 1-5):

1. DOCTRINAL ACCURACY (weight: {EVAL_WEIGHTS['doctrinal_accuracy']:.0%} — MOST IMPORTANT)
   This is the non-negotiable dimension. Any score below 5 must include specific examples.
   - 5: Perfect preservation of ALL doctrinal content, nuance, and implication.
        Every theological claim, condemnation, and distinction is intact.
   - 4: Core doctrine preserved, but 1-2 very minor non-doctrinal imprecisions
        (e.g., slightly awkward phrasing that doesn't change meaning).
   - 3: Generally accurate, but some theological nuances lost or slightly shifted.
        A careful reader would notice differences from the original.
   - 2: Significant doctrinal content lost, added, or distorted. Some theological
        claims are materially different from the original.
   - 1: Core theological meaning is wrong or fundamentally changed. This translation
        would teach false doctrine if used in a church setting.

2. TERMINOLOGY CONSISTENCY (weight: {EVAL_WEIGHTS['terminology_consistency']:.0%})
   Key theological terms must be translated the same way every time they appear.
   - 5: All key terms use standard, established theological vocabulary for the
        target language. Perfect consistency throughout the passage.
   - 4: Nearly all terms correct and consistent; one very minor inconsistency
        that doesn't affect understanding.
   - 3: Some important terms inconsistent or using non-standard translations.
        A theological reader would notice the inconsistency.
   - 2: Multiple key theological terms wrong, inconsistent, or using confusing
        vocabulary that could mislead readers.
   - 1: Theological terminology is largely incorrect, random, or inconsistent
        to the point of being unreliable.

3. CLARITY (weight: {EVAL_WEIGHTS['clarity']:.0%})
   Would a native speaker of the target language understand this clearly?
   - 5: Perfectly clear and understandable to any educated native speaker.
        Meaning is immediately apparent on first reading.
   - 4: Clear with perhaps one slightly awkward construction that requires
        a moment's pause but doesn't obscure meaning.
   - 3: Understandable but requires effort or re-reading in places.
        Some sentences are confusing on first read.
   - 2: Confusing in multiple places. A reader would struggle to follow
        the argument or identify the theological claims.
   - 1: Largely incomprehensible to a native speaker. The text fails to
        communicate its content.

4. NATURALNESS (weight: {EVAL_WEIGHTS['naturalness']:.0%} — lowest priority)
   Does this read like natural prose in the target language?
   Note: for confessional texts, a formal/elevated register is expected and appropriate.
   - 5: Reads as if originally composed in the target language. Natural word
        order, appropriate register for theological prose.
   - 4: Mostly natural with minor traces of translation (e.g., slightly
        unusual word order in one place).
   - 3: Clearly a translation but acceptable for its purpose. Some sentences
        feel constructed rather than natural.
   - 2: Awkward and stilted. Obviously machine-translated or overly literal.
        Would distract a reader from the content.
   - 1: Unnatural to the point of being painful to read. Grammar or word
        choices that no native speaker would use.

RESPONSE FORMAT:
You MUST respond with valid JSON only. No markdown code blocks, no explanations
outside the JSON. Use this EXACT structure:
{{
  "doctrinal_accuracy": {{
    "score": <1-5>,
    "explanation": "<MUST cite specific phrases from both source and translation as evidence>"
  }},
  "terminology_consistency": {{
    "score": <1-5>,
    "explanation": "<MUST list specific terms checked and whether they were consistent>"
  }},
  "clarity": {{
    "score": <1-5>,
    "explanation": "<MUST identify any unclear passages with specific quotes>"
  }},
  "naturalness": {{
    "score": <1-5>,
    "explanation": "<MUST give examples of natural OR unnatural constructions>"
  }},
  "critical_errors": [
    "<list ONLY places where doctrinal meaning was changed, lost, or added — empty array if none>"
  ],
  "suggestions": [
    "<specific, actionable suggestions for improving the translation>"
  ]
}}"""


# =========================================================================
# RUBRIC EVALUATION (with dual-pass consistency)
# =========================================================================

def _run_single_rubric(client, source_text, translated_text, target_language,
                       glossary=None, variant="A"):
    """Run a single rubric evaluation pass."""
    user_message = f"""Evaluate this translation from {SOURCE_LANGUAGE} to {target_language}.

=== SOURCE TEXT ({SOURCE_LANGUAGE}) ===
{source_text}

=== TRANSLATION ({target_language}) ===
{translated_text}"""

    if glossary:
        glossary_text = "\n".join(
            f"  {src} -> {tgt}"
            for src, tgt in glossary.items()
        )
        user_message += f"""

=== APPROVED GLOSSARY TERMS ===
The following are the approved translations for key theological terms.
Check whether the translation uses these consistently:
{glossary_text}"""

    response = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=MAX_TOKENS,
        system=_build_rubric_prompt(variant),
        messages=[{"role": "user", "content": user_message}],
    )

    response_text = response.content[0].text.strip()
    parsed = _parse_json_response(response_text)

    # If parsing failed, ask the AI to reformat its own output.
    # This handles ANY malformed response — truncation, weird chars,
    # markdown wrapping, anything we haven't seen yet.
    if "error" in parsed:
        print("    [JSON recovery] Parse failed, asking AI to re-extract...")

        # First, pull the scores from the raw text with regex so we can
        # verify the AI doesn't change them during reformatting
        raw_scores = _extract_scores_from_raw(response_text)

        recovery = client.messages.create(
            model=EVAL_MODEL,
            max_tokens=MAX_TOKENS,
            system=(
                "CONTEXT: You are a JSON recovery tool inside Step 5 (Evaluation) "
                "of a WELS Lutheran translation pipeline. A previous AI evaluator "
                "produced a rubric evaluation but the response was not valid JSON. "
                "Your ONLY job is to reformat the existing evaluation as clean JSON.\n\n"
                "CRITICAL RULES:\n"
                "- You are NOT an evaluator. Do NOT re-evaluate anything.\n"
                "- Copy every score EXACTLY as the number that appears in the text.\n"
                "- Copy every explanation as-is. Do not reword, shorten, or expand.\n"
                "- Copy every critical_error and suggestion verbatim.\n"
                "- If a score says 3, output 3. If it says 4, output 4. No changes.\n"
                "- Output ONLY the JSON object. No markdown, no commentary."
            ),
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": (
                    "Your evaluation response above could not be parsed as JSON. "
                    "Please reformat it as a clean JSON object. COPY every value "
                    "exactly as it appeared — same scores, same explanations, same "
                    "errors, same suggestions. Do not re-evaluate or adjust anything. "
                    "Output ONLY the JSON with keys: doctrinal_accuracy, "
                    "terminology_consistency, clarity, naturalness (each with score "
                    "and explanation), critical_errors (array), suggestions (array)."
                )},
            ],
        )
        recovery_text = recovery.content[0].text.strip()
        parsed = _parse_json_response(recovery_text)

        # Verify the recovery didn't change the scores
        if "error" not in parsed and raw_scores:
            parsed = _verify_recovered_scores(parsed, raw_scores, response_text)

    return parsed


def evaluate_with_rubric(source_text, translated_text, target_language, glossary=None):
    """
    Dual-pass rubric evaluation for consistency.

    Runs the rubric evaluation TWICE with different prompt framings.
    If scores agree (within 1 point on all dimensions), averages them.
    If scores disagree on any dimension, runs a third tiebreaker pass
    and takes the median (middle) score for that dimension.

    Returns the final reconciled scores plus a consistency report.
    """
    client = get_client()
    dimensions = ["doctrinal_accuracy", "terminology_consistency", "clarity", "naturalness"]

    # Pass A
    print("    [Rubric pass A]...")
    pass_a = _run_single_rubric(client, source_text, translated_text,
                                target_language, glossary, "A")
    if "error" in pass_a:
        return pass_a

    # Pass B
    print("    [Rubric pass B]...")
    pass_b = _run_single_rubric(client, source_text, translated_text,
                                target_language, glossary, "B")
    if "error" in pass_b:
        return pass_b

    # Check consistency
    disagreements = []
    for dim in dimensions:
        score_a = pass_a.get(dim, {}).get("score", 0)
        score_b = pass_b.get(dim, {}).get("score", 0)
        if abs(score_a - score_b) > 1:
            disagreements.append(dim)

    # If disagreements exist, run tiebreaker
    pass_c = None
    if disagreements:
        print(f"    [Tiebreaker pass — disagreement on: {', '.join(disagreements)}]...")
        pass_c = _run_single_rubric(client, source_text, translated_text,
                                    target_language, glossary, "A")

    # Build final reconciled result
    result = {}
    consistency_report = {}

    for dim in dimensions:
        scores = [
            pass_a.get(dim, {}).get("score", 0),
            pass_b.get(dim, {}).get("score", 0),
        ]
        explanations = [
            pass_a.get(dim, {}).get("explanation", ""),
            pass_b.get(dim, {}).get("explanation", ""),
        ]

        if pass_c and dim in disagreements:
            scores.append(pass_c.get(dim, {}).get("score", 0))
            explanations.append(pass_c.get(dim, {}).get("explanation", ""))

        # Final score: median (middle value) of all passes
        sorted_scores = sorted(scores)
        if len(sorted_scores) == 3:
            final_score = sorted_scores[1]  # true median
        else:
            final_score = round(sum(sorted_scores) / len(sorted_scores), 1)

        # Use the explanation from whichever pass gave the final score
        # (prefer the longer, more detailed one)
        best_explanation = max(explanations, key=len)

        result[dim] = {
            "score": final_score,
            "explanation": best_explanation,
        }

        consistency_report[dim] = {
            "all_scores": scores,
            "agreed": abs(scores[0] - scores[1]) <= 1,
            "final_score": final_score,
            "tiebreaker_used": pass_c is not None and dim in disagreements,
        }

    # Merge critical errors and suggestions from all passes
    all_errors = []
    all_suggestions = []
    for p in [pass_a, pass_b] + ([pass_c] if pass_c else []):
        all_errors.extend(p.get("critical_errors", []))
        all_suggestions.extend(p.get("suggestions", []))

    # Deduplicate by keeping unique non-empty entries
    result["critical_errors"] = list(set(e for e in all_errors if e))
    result["suggestions"] = list(set(s for s in all_suggestions if s))

    # Calculate weighted score
    weighted_score = sum(
        result[dim]["score"] * EVAL_WEIGHTS[dim]
        for dim in dimensions
    )
    result["weighted_score"] = round(weighted_score, 2)
    result["max_possible_score"] = 5.0
    result["consistency_report"] = consistency_report

    return result


# =========================================================================
# BACK-TRANSLATION
# =========================================================================

def back_translate(translated_text, source_language, target_language):
    """
    Translates the translated text BACK into the source language.
    Uses a deliberately literal approach so that any meaning shifts
    in the translation are exposed in the back-translated version.
    """
    client = get_client()

    response = client.messages.create(
        model=BACK_TRANSLATE_MODEL,
        max_tokens=MAX_TOKENS,
        system=f"""PIPELINE CONTEXT: You are Step 5b in a WELS Lutheran translation pipeline.
Before you, an AI translator translated a {source_language} confessional text into
{target_language}. Your job is to translate that {target_language} text BACK into
{source_language}. After you, a comparison AI will place your back-translation next
to the original and look for any meaning differences. If you "fix" or "improve" the
text, the comparison step won't catch real translation errors — so your literal
faithfulness is essential to the quality of the entire pipeline.

YOUR ROLE: You are ONLY a back-translator for quality assurance. You are NOT
improving, fixing, or editing anything. You are creating a mirror that reveals
exactly what the {target_language} text says.

CRITICAL INSTRUCTIONS:
- Translate as LITERALLY and FAITHFULLY as possible
- Do NOT try to improve, smooth, or embellish the text
- Do NOT try to guess what the original {source_language} said — translate what IS there
- If the {target_language} text is awkward, your back-translation MUST be equally awkward
- If the {target_language} text has an error, your back-translation MUST reflect that error
- We need to see EXACTLY what the {target_language} text communicates,
  even if it sounds unnatural in {source_language}
- Do NOT add words, remove words, or rephrase. Translate what exists.

Respond with ONLY the translation. No explanations, notes, or commentary.""",
        messages=[{
            "role": "user",
            "content": f"Translate this {target_language} text into {source_language}:\n\n{translated_text}"
        }],
    )

    return response.content[0].text


def compare_with_original(original_text, back_translated_text, source_language):
    """
    Compares the original source with the back-translated version to detect
    meaning loss or distortion. Specifically trained to catch doctrinal shifts.
    """
    client = get_client()

    response = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=MAX_TOKENS,
        system=f"""PIPELINE CONTEXT: You are part of Step 5 (Evaluation) in a WELS
Lutheran translation pipeline. Here is what happened before you:
  - Step 4: An AI translator produced a translation from {source_language} to a target language.
  - Step 5a: An AI evaluator scored the translation on a rubric.
  - Step 5b: A DIFFERENT AI back-translated the translation into {source_language}.
  - Step 5c (YOU): You compare the original with the back-translation to catch meaning loss.

Your comparison results will be included in an evaluation report that humans review,
and may also be fed back into Step 6 (Improvement) to fix specific issues. Your
severity ratings directly influence what gets flagged for fixes.

YOUR ROLE: You are ONLY a comparator. You do NOT translate, improve, or fix anything.
You identify differences and rate their severity honestly. Do NOT inflate or deflate
severity ratings. Report exactly what you see.

You will compare two versions of a text in {source_language}:
1. The ORIGINAL text
2. A BACK-TRANSLATED version (translated to another language and back)

Your job: identify every difference in MEANING between them.

RULES:
- IGNORE minor stylistic differences, word-order changes, or synonym substitutions
  that do not change the theological meaning.
- FOCUS on theological content. Any shift in doctrinal meaning is CRITICAL, even if
  the phrasing sounds similar.
- Pay special attention to: condemnation clauses, attribution of actions to God vs.
  humans, descriptions of sacraments, and statements about justification/salvation.
- A "critical" difference is one that would change what a church teaches.
- A "moderate" difference is one a theologian would notice but wouldn't change doctrine.
- A "minor" difference is purely stylistic.

Respond with valid JSON only (no markdown, no code blocks):
{{
  "meaning_preserved": <true or false — overall, was the core DOCTRINAL meaning preserved?>,
  "differences": [
    {{
      "original_phrase": "<exact phrase from the original>",
      "back_translated_phrase": "<corresponding phrase from the back-translation>",
      "severity": "<critical|moderate|minor>",
      "explanation": "<what meaning changed and WHY it matters doctrinally>"
    }}
  ],
  "overall_assessment": "<brief summary of how well meaning was preserved>"
}}""",
        messages=[{
            "role": "user",
            "content": f"""Compare these two {source_language} texts:

=== ORIGINAL ===
{original_text}

=== BACK-TRANSLATED ===
{back_translated_text}"""
        }],
    )

    response_text = response.content[0].text.strip()
    parsed = _parse_json_response(response_text)

    # Same recovery pattern: if parsing fails, ask AI to reformat
    if "error" in parsed:
        print("    [JSON recovery] Comparison parse failed, asking AI to re-extract...")
        recovery = client.messages.create(
            model=EVAL_MODEL,
            max_tokens=MAX_TOKENS,
            system=(
                "CONTEXT: You are a JSON recovery tool inside Step 5 (Evaluation) "
                "of a WELS Lutheran translation pipeline. A previous AI compared "
                "an original source text with a back-translated version to check "
                "for meaning loss, but its response was not valid JSON. "
                "Your ONLY job is to reformat it as clean JSON.\n\n"
                "CRITICAL RULES:\n"
                "- You are NOT a comparator. Do NOT re-analyze the texts.\n"
                "- Copy meaning_preserved EXACTLY as stated (true or false).\n"
                "- Copy every difference with its exact severity (critical/moderate/minor).\n"
                "- Do NOT upgrade or downgrade any severity level.\n"
                "- Copy every phrase and explanation verbatim.\n"
                "- Output ONLY the JSON object. No markdown, no commentary."
            ),
            messages=[
                {"role": "user", "content": f"Compare these two texts:\n\nORIGINAL:\n{original_text}\n\nBACK-TRANSLATED:\n{back_translated_text}"},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": (
                    "Your comparison response above could not be parsed as JSON. "
                    "Please reformat it as a clean JSON object. COPY every value "
                    "exactly — same meaning_preserved, same severities, same phrases, "
                    "same explanations. Do not re-analyze. Output ONLY the JSON with "
                    "keys: meaning_preserved (bool), differences (array of objects "
                    "with original_phrase, back_translated_phrase, severity, explanation), "
                    "overall_assessment (string)."
                )},
            ],
        )
        parsed = _parse_json_response(recovery.content[0].text.strip())

    return parsed


# =========================================================================
# FULL EVALUATION (combines all methods)
# =========================================================================

def full_evaluation(source_text, translated_text, target_language,
                    chunk_id=None, glossary=None, source_label=None):
    """
    Runs the complete evaluation: dual-pass rubric + back-translation + comparison.

    Parameters:
        source_text (str): The original text in the source language
        translated_text (str): The translation to evaluate
        target_language (str): Name of the target language (e.g., "Hungarian")
        chunk_id (str, optional): Identifier for this chunk
        glossary (dict, optional): {source_term: target_term} dictionary
        source_label (str, optional): Label for what produced this translation
                                      (e.g., "pipeline_v1", "benjamin_pass2", "manual")

    Returns:
        dict: Complete evaluation with rubric scores, back-translation, and consistency report
    """
    print(f"  [1/4] Running rubric evaluation (dual-pass)...")
    rubric_result = evaluate_with_rubric(
        source_text, translated_text, target_language, glossary
    )

    print(f"  [2/4] Back-translating to {SOURCE_LANGUAGE}...")
    back_translated = back_translate(
        translated_text, SOURCE_LANGUAGE, target_language
    )

    print(f"  [3/4] Comparing back-translation with original...")
    comparison = compare_with_original(
        source_text, back_translated, SOURCE_LANGUAGE
    )

    print(f"  [4/4] Assembling final evaluation...")

    result = {
        "chunk_id": chunk_id,
        "target_language": target_language,
        "source_label": source_label or "unknown",
        "rubric_evaluation": rubric_result,
        "back_translation": {
            "back_translated_text": back_translated,
            "comparison": comparison,
        },
    }

    return result
