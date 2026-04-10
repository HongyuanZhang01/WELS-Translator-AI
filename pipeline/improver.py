"""
improver.py - Step 6: Context-Aware Improvement with Verification

This is the third part of the translate -> evaluate -> improve loop.
It takes a translation that didn't score 5/5, reads the evaluator's
specific feedback, and sends it back to the AI for targeted fixes.

KEY DESIGN: The improver gets the SAME context the original translator
had (glossary, document context, surrounding chunks, theological rules)
PLUS the evaluation feedback. This means it has every resource needed
to make genuinely good fixes, not just blind patches.

VERIFICATION SYSTEM:
  After each improvement attempt, we verify:
    1. GLOSSARY CHECK: Are all approved terms still present?
    2. SCORE CHECK: Did the scores go up, not down?
  If either check fails, we reject the improvement and try again with
  added context about WHY the previous attempt failed — so each retry
  is smarter than the last.

INPUT:  Original source + current translation + evaluation + full context
OUTPUT: A verified improved translation (or the original if no improvement worked)
"""

import json
import os
import re
import sys
from anthropic import Anthropic

sys.path.insert(0, os.path.dirname(__file__))
from config import TRANSLATE_MODEL, MAX_TOKENS, SOURCE_LANGUAGE
# call_with_retry wraps client.messages.create() with automatic
# retry-on-transient-failure (500/503/429/network errors).
from api_retry import call_with_retry


def get_client():
    return Anthropic()


# =========================================================================
# SYSTEM PROMPT — matches the translator's theological depth + revision focus
# =========================================================================

def build_improvement_prompt(source_language, target_language, document_context=None):
    """
    System prompt for the improvement pass. This mirrors the translator's
    full theological instructions but adds revision-specific rules.
    """
    context_section = ""
    if document_context:
        context_section = f"""
DOCUMENT CONTEXT:
{document_context}
"""

    return f"""You are an expert translator of Lutheran confessional documents,
performing a TARGETED REVISION pass on an existing translation from
{source_language} to {target_language} for the Wisconsin Evangelical
Lutheran Synod (WELS).
{context_section}
You have the same theological expertise as the original translator.
You will receive:
  1. The approved glossary of theological terms
  2. The original source text
  3. The current translation (which is mostly good but has specific issues)
  4. Specific issues identified by a quality evaluator
  5. Possibly: context from previous failed fix attempts

YOUR MISSION: Fix ONLY the flagged issues. The current translation is your
starting point — it has been carefully produced and most of it is correct.
Your job is surgical: fix what's broken, leave everything else UNTOUCHED.

ABSOLUTE RULES:

1. GLOSSARY TERMS ARE SACRED.
   Every glossary term that currently appears correctly in the translation
   MUST remain exactly as-is. If a glossary term is WRONG, replace it with
   the approved term. If it's RIGHT, do NOT touch it.
   Before outputting, mentally verify every glossary term is present.

2. DOCTRINAL PRECISION IS NON-NEGOTIABLE.
   Every theological concept must be translated with its exact meaning.
   Do not soften, generalize, or modernize doctrinal language.

3. FIX ONLY WHAT'S FLAGGED.
   Do not rewrite the entire translation. Do not "improve" unflagged
   sections. If a sentence was not mentioned in the issues list, output
   it EXACTLY as it appears in the current translation — character for
   character, word for word.

4. PRESERVE THE AUTHOR'S STYLE.
   Keep the original tone, rhetorical force, and sentence structure.
   If the original is repetitive, keep it repetitive. If it's harsh,
   keep it harsh.

5. PRESERVE STRUCTURE.
   Maintain paragraph breaks, numbered points, and section markers.

6. DO NOT ADD OR REMOVE CONTENT.
   Do not add explanatory notes, footnotes, or translator comments.
   Every sentence in the source must have a corresponding sentence.

7. SCRIPTURE REFERENCES.
   Use the standard {target_language} Bible translation for quoted passages.

8. LATIN/GREEK PHRASES.
   Keep them in Latin/Greek with a {target_language} translation in
   parentheses on first occurrence only.

9. IF A SUGGESTION CONFLICTS WITH DOCTRINE, PRIORITIZE DOCTRINE.

OUTPUT: Respond with ONLY the revised translation text. No explanations,
no notes about what you changed. Just the improved text."""


# =========================================================================
# GLOSSARY VERIFICATION
# =========================================================================

def verify_glossary_terms(text, glossary):
    """
    Checks which approved glossary terms appear in the text.
    Returns a dict with:
      - present: list of terms found
      - missing: list of terms that should be there but aren't
      - term_map: {source_term: target_term} for all checked terms
    """
    if not glossary:
        return {"present": [], "missing": [], "term_map": {}}

    present = []
    missing = []

    for source_term, target_term in glossary.items():
        if target_term and target_term in text:
            present.append({"source": source_term, "target": target_term})
        # Only flag as missing if the source concept is likely relevant
        # (we can't check all 200+ glossary terms — just the ones whose
        # target translations were in the PREVIOUS version)

    return {"present": present, "missing": missing, "term_map": glossary}


def find_glossary_regressions(original_text, improved_text, glossary):
    """
    Compares the original and improved texts to find glossary terms that
    were PRESENT in the original but DISAPPEARED in the improved version.
    These are regressions — the improver broke something that was working.

    Returns:
      list of dicts: [{source_term, target_term, issue}]
    """
    if not glossary:
        return []

    regressions = []
    for source_term, target_term in glossary.items():
        if not target_term:
            continue
        was_present = target_term in original_text
        still_present = target_term in improved_text
        if was_present and not still_present:
            regressions.append({
                "source_term": source_term,
                "target_term": target_term,
                "issue": f"Glossary term '{target_term}' (for '{source_term}') "
                         f"was present in the original but disappeared after improvement"
            })

    return regressions


# =========================================================================
# ISSUE EXTRACTION FROM EVALUATION
# =========================================================================

def extract_issues(eval_result):
    """
    Pulls out the specific issues from an evaluation result into a
    structured, prioritized list.
    """
    rubric = eval_result.get("rubric_evaluation", {})
    bt = eval_result.get("back_translation", {}).get("comparison", {})
    issues = []

    # Critical errors (highest priority)
    for err in rubric.get("critical_errors", []):
        if err:
            issues.append(f"CRITICAL ERROR: {err}")

    # Dimension-specific feedback for any dimension below 5
    dimensions = [
        "doctrinal_accuracy", "terminology_consistency",
        "clarity", "naturalness"
    ]
    for dim in dimensions:
        dim_data = rubric.get(dim, {})
        score = dim_data.get("score", 5)
        if score < 5:
            explanation = dim_data.get("explanation", "")
            issues.append(f"{dim.upper()} (scored {score}/5): {explanation}")

    # Suggestions
    for sug in rubric.get("suggestions", []):
        if sug:
            issues.append(f"SUGGESTION: {sug}")

    # Back-translation differences (moderate and critical only)
    for diff in bt.get("differences", []):
        severity = diff.get("severity", "minor")
        if severity in ("critical", "moderate"):
            issues.append(
                f"BACK-TRANSLATION ({severity.upper()}): "
                f"'{diff.get('original_phrase', '')}' became "
                f"'{diff.get('back_translated_phrase', '')}' — "
                f"{diff.get('explanation', '')}"
            )

    return issues


# =========================================================================
# CORE IMPROVEMENT FUNCTION
# =========================================================================

def improve_translation(source_text, translated_text, eval_result,
                        target_language, glossary=None,
                        source_language=None, document_context=None,
                        context_before=None, context_after=None,
                        previous_failures=None):
    """
    Produces an improved translation using the FULL context the original
    translator had, plus evaluation feedback and failure history.

    Parameters:
        source_text (str): The original source text
        translated_text (str): The current translation to improve
        eval_result (dict): The evaluation result from evaluator.py
        target_language (str): Target language name
        glossary (dict): {source_term: target_term} approved terms
        source_language (str): Source language override
        document_context (str): Document context (same as translator had)
        context_before (str): Preceding chunk text (same as translator had)
        context_after (str): Following chunk text (same as translator had)
        previous_failures (list): List of dicts describing why previous
            improvement attempts were rejected. Each dict has:
            {attempt: int, improved_text: str, rejection_reason: str,
             glossary_regressions: list}

    Returns:
        str: The improved translation text
    """
    lang = source_language or SOURCE_LANGUAGE
    client = get_client()

    # Build system prompt with full translator context
    system = build_improvement_prompt(lang, target_language, document_context)

    # Extract issues from evaluation
    issues = extract_issues(eval_result)
    issues_text = "\n\n".join(f"{i+1}. {issue}" for i, issue in enumerate(issues))

    # Build the user message with full context (mirroring translator.py)
    user_parts = []

    # 1. Glossary (same as translator provides)
    if glossary:
        glossary_text = "\n".join(
            f"  {src} = {tgt}" for src, tgt in glossary.items()
        )
        user_parts.append(
            "APPROVED GLOSSARY — You MUST use these exact terms. "
            "Any glossary term that is ALREADY CORRECT in the current "
            "translation must remain UNCHANGED:\n" + glossary_text
        )

    # 2. Surrounding context (same as translator had)
    if context_before:
        user_parts.append(
            f"PRECEDING CONTEXT (for reference, do NOT translate):\n"
            f"...{context_before}"
        )

    # 3. Original source text
    user_parts.append(f"ORIGINAL SOURCE TEXT ({lang}):\n{source_text}")

    if context_after:
        user_parts.append(
            f"FOLLOWING CONTEXT (for reference, do NOT translate):\n"
            f"{context_after}..."
        )

    # 4. Current translation (the starting point for revision)
    user_parts.append(
        f"CURRENT TRANSLATION ({target_language}) — this is your "
        f"starting point. Fix ONLY the issues listed below:\n{translated_text}"
    )

    # 5. Issues to fix
    user_parts.append(
        f"ISSUES TO FIX (address ALL of these):\n{issues_text}"
    )

    # 6. Previous failure context (if any)
    if previous_failures:
        failure_text = []
        for fail in previous_failures:
            attempt = fail.get("attempt", "?")
            reason = fail.get("rejection_reason", "unknown")
            regressions = fail.get("glossary_regressions", [])

            entry = f"ATTEMPT {attempt} WAS REJECTED: {reason}"
            if regressions:
                reg_details = "\n".join(
                    f"  - {r['issue']}" for r in regressions
                )
                entry += f"\n  Glossary terms that were broken:\n{reg_details}"
            failure_text.append(entry)

        user_parts.append(
            "WARNING — PREVIOUS FIX ATTEMPTS FAILED. Learn from these "
            "mistakes and do NOT repeat them:\n" +
            "\n\n".join(failure_text)
        )

    # Join with clear separators
    user_message = ("\n\n" + "=" * 50 + "\n\n").join(user_parts)

    try:
        response = call_with_retry(
            client,
            model=TRANSLATE_MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        improved = response.content[0].text.strip()
        if not improved:
            print("    WARNING: Improver returned empty text, keeping original")
            return translated_text
        return improved
    except Exception as e:
        print(f"    WARNING: Improvement API call failed: {e}")
        print(f"    Keeping the current translation and continuing...")
        return translated_text


# =========================================================================
# VERIFIED IMPROVEMENT — the main entry point for the pipeline
# =========================================================================

def verified_improve(source_text, translated_text, eval_result,
                     target_language, glossary=None,
                     source_language=None, document_context=None,
                     context_before=None, context_after=None,
                     max_attempts=2):
    """
    Attempts to improve a translation with VERIFICATION after each attempt.
    If the improvement introduces glossary regressions, it retries with
    knowledge of what went wrong.

    Parameters:
        (same as improve_translation, plus:)
        max_attempts (int): Max fix attempts per improvement loop (default: 2)

    Returns:
        dict: {
            "improved_text": str,     # the best translation we got
            "accepted": bool,         # True if improvement was accepted
            "attempts": int,          # how many attempts were made
            "rejection_reasons": list  # why any attempts were rejected
        }
    """
    previous_failures = []
    best_text = translated_text

    for attempt in range(1, max_attempts + 1):
        print(f"    [attempt {attempt}/{max_attempts}] Generating improvement...")

        improved_text = improve_translation(
            source_text=source_text,
            translated_text=translated_text,  # Always start from the ORIGINAL
            eval_result=eval_result,
            target_language=target_language,
            glossary=glossary,
            source_language=source_language,
            document_context=document_context,
            context_before=context_before,
            context_after=context_after,
            previous_failures=previous_failures if previous_failures else None,
        )

        # If improver returned the original text (error fallback), stop
        if improved_text == translated_text:
            print(f"    [attempt {attempt}] No changes produced, keeping original")
            return {
                "improved_text": translated_text,
                "accepted": False,
                "attempts": attempt,
                "rejection_reasons": ["Improver returned unchanged text"],
            }

        # VERIFICATION GATE 1: Check glossary regressions
        regressions = find_glossary_regressions(
            translated_text, improved_text, glossary
        )

        if regressions:
            reason = (
                f"Glossary regression: {len(regressions)} approved term(s) "
                f"were removed or changed"
            )
            print(f"    [attempt {attempt}] REJECTED — {reason}")
            for reg in regressions:
                print(f"      - {reg['issue']}")

            previous_failures.append({
                "attempt": attempt,
                "improved_text": improved_text,
                "rejection_reason": reason,
                "glossary_regressions": regressions,
            })
            continue  # Try again with failure context

        # Passed all checks — accept this improvement
        print(f"    [attempt {attempt}] ACCEPTED — glossary verified")
        return {
            "improved_text": improved_text,
            "accepted": True,
            "attempts": attempt,
            "rejection_reasons": [],
        }

    # All attempts exhausted — keep the original
    print(f"    All {max_attempts} attempts rejected, keeping original translation")
    return {
        "improved_text": translated_text,
        "accepted": False,
        "attempts": max_attempts,
        "rejection_reasons": [f.get("rejection_reason", "") for f in previous_failures],
    }


# =========================================================================
# NEEDS IMPROVEMENT CHECK
# =========================================================================

def needs_improvement(eval_result, min_doctrinal=5, min_terminology=5):
    """
    Check whether this evaluation result indicates the translation
    needs another improvement pass.

    Returns True if doctrinal_accuracy or terminology_consistency
    is below the minimum threshold.

    If the evaluation itself errored (e.g., rubric JSON unparseable
    even after in-pass recovery AND Pass-B fallback), returns True
    to force an improvement attempt rather than silently treating a
    failed eval as a pass. The alternative — returning False — caused
    Hungarian chunk 4 in the 2026-04-09 wauwatosa run to be reported
    as "TARGET MET" with 0/0/0 scores. Forcing improvement means the
    worst case is we waste one improvement cycle; the best case is
    we rescue a translation that would otherwise have silently
    received a fake pass.
    """
    rubric = eval_result.get("rubric_evaluation", {})
    if "error" in rubric:
        print("  [WARNING] Rubric evaluation errored — forcing improvement attempt "
              "rather than treating failed eval as TARGET MET.")
        return True  # force improvement when we cannot trust the eval

    da = rubric.get("doctrinal_accuracy", {}).get("score", 0)
    tc = rubric.get("terminology_consistency", {}).get("score", 0)

    return da < min_doctrinal or tc < min_terminology
