import json, os, csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-not-needed")
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"

INPUT       = os.path.expanduser("~/ge_rig/output/theological_vocabulary.json")
OUTPUT_JSON = os.path.expanduser("~/ge_rig/output/theological_vocabulary_deduped.json")
OUTPUT_CSV  = os.path.expanduser("~/ge_rig/output/theological_vocabulary_deduped.csv")
BATCH_SIZE  = 40  # tune to taste

SYSTEM_MERGE = """\
You are a Lutheran theological lexicographer consolidating duplicate glossary entries.
You will receive multiple entries for the same German term, drawn from different chunks \
of C. F. W. Walther's *The Proper Distinction Between Law and Gospel*.

Your task: synthesize all variants into a single canonical entry that:
- Preserves ALL distinct nuances and meanings across variants
- Resolves contradictions by choosing the most precise Lutheran formulation
- Produces the most complete and useful translation_note for a Hungarian translator
- Does not simply pick one entry and discard the others

Return ONLY valid JSON with exactly these keys:
{
  "term": "<German term>",
  "english": "<English gloss>",
  "lutheran_nuance": "<synthesized nuance, 15-30 words>",
  "translation_note": "<synthesized translator guidance, 15-25 words>"
}

No prose outside the JSON object. No markdown fences."""


def merge_entries(term, variants):
    variants_text = json.dumps(variants, ensure_ascii=False, indent=2)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_MERGE},
            {"role": "user",   "content": f"Merge these entries for '{term}':\n\n{variants_text}"}
        ],
        temperature=0.1,
        max_tokens=512,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        },
        timeout=120,
    )

    raw = response.choices[0].message.content.strip()
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def process_term(term, variants):
    """Worker function — returns (term, merged_entry, error_or_None)."""
    if len(variants) == 1:
        entry = {k: v for k, v in variants[0].items() if k != "chunk"}
        return term, entry, None

    clean_variants = [{k: v for k, v in e.items() if k != "chunk"}
                      for e in variants]
    try:
        merged = merge_entries(term, clean_variants)
        return term, merged, None
    except Exception as e:
        fallback = {k: v for k, v in variants[0].items() if k != "chunk"}
        return term, fallback, str(e)


def main():
    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    # Group by term
    grouped = defaultdict(list)
    for entry in data:
        term = entry.get("term", "").strip()
        if term:
            grouped[term].append(entry)

    total  = len(grouped)
    singles = sum(1 for v in grouped.values() if len(v) == 1)
    multi   = total - singles
    print(f"Total unique terms: {total} ({singles} single, {multi} need merging)")
    print(f"Model: {MODEL} | Batch size: {BATCH_SIZE}\n")

    # Results dict preserves nothing about order until we sort at the end
    results = {}
    errors  = []

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = {
            executor.submit(process_term, term, variants): term
            for term, variants in grouped.items()
        }

        with tqdm(total=total, unit="term") as pbar:
            for future in as_completed(futures):
                term = futures[future]
                try:
                    term, entry, error = future.result()
                    results[term] = entry
                    if error:
                        errors.append((term, error))
                        print(f"\n  [fallback] '{term}': {error}")
                except Exception as e:
                    print(f"\n  [unhandled] '{term}': {e}")
                    errors.append((term, str(e)))
                pbar.update(1)

    # Sort alphabetically for consistent output
    output = [results[term] for term in sorted(results.keys())]

    # Write JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Write CSV
    fields = ["term", "english", "lutheran_nuance", "translation_note"]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)

    print(f"\nDone: {len(output)} canonical terms written")
    if errors:
        print(f"Errors (fell back to first variant): {len(errors)}")
        for t, e in errors:
            print(f"  {t}: {e}")


if __name__ == "__main__":
    main()
