"""
chunker.py - Step 1: Document Ingestion and Chunking

Takes a source document and splits it into overlapping chunks while
preserving document structure (articles, chapters, sections).

WHY CHUNKING MATTERS:
  AI models can only process a limited amount of text at once. If you
  feed it a 100-page book, it can't handle it. So we split the text
  into smaller pieces ("chunks") that the AI can translate one at a time.

WHY OVERLAP MATTERS:
  If you cut the text at exactly 500 words, you might split a sentence
  in half, or the AI won't know what "this doctrine" refers to because
  the explanation was in the previous chunk. Overlap means each chunk
  includes a bit of the previous chunk and a bit of the next one, so
  the translator always has context.

WHY STRUCTURE MATTERS:
  If the AI knows "this chunk is from Article IV: Justification," it
  can make much better translation choices than if it just sees a block
  of text with no idea where it came from.

INPUT:  Plain text file (.txt) or extracted PDF text
OUTPUT: A JSON file with chunks, each containing:
        - chunk_id: unique identifier
        - text: the chunk content
        - context_before: overlap text from the previous chunk
        - context_after: overlap text from the next chunk
        - article: which article/section this belongs to (if detected)
        - position: where in the document this chunk falls
"""

import json
import re
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from config import SOURCE_LANGUAGE


# =========================================================================
# STRUCTURE DETECTION
# =========================================================================

def detect_articles(text):
    """
    Scans the text and finds article/section boundaries.

    Returns a list of dicts:
      [{"title": "Article 1: God", "start": 0, "end": 450}, ...]

    Supports multiple formats:
      - "Article 1, God" / "Article I: Of God"
      - "Thesis I." / "Thesis XIV."
      - "Chapter 1" / "Part One"
      - Numbered sections like "[1]" at the start of paragraphs
    """
    patterns = [
        # "Article 1, God" or "Article IV: Justification"
        r'(?m)^(Article\s+[IVXLCDM\d]+[.:,]\s*.+?)$',
        # "Articulus I: De Deo" (Latin)
        r'(?m)^(Articulus\s+[IVXLCDM\d]+[.:,]?\s*.*)$',
        # "Thesis I." style (from Walther)
        r'(?m)^(Thesis\s+[IVXLCDM\d]+\.?\s*.*)$',
        # "Chapter 1" or "Part One"
        r'(?m)^((?:Chapter|Part|Section)\s+[\w\d]+[.:,]?\s*.*)$',
        # German catechism headers: "DAS ERSTE GEBOT", "DER ZWEITE ARTIKEL",
        # "DIE ERSTE BITTE", "EINE CHRISTLICHE ... VORREDE", etc.
        r'(?m)^((?:DAS|DER|DIE|EINE|VON)\s+[A-ZÄÖÜ].{3,60})$',
        # German part headers: "Der erste Teil", "Der zweite Teil", "Der dritte Teil"
        r'(?m)^(Der\s+(?:erste|zweite|dritte|vierte|fünfte)\s+Teil)$',
        # Generic numbered headers like "I. Of God", "II. Of Sin", "IV. Justification"
        # IMPORTANT: we require the line to start with I/V/X/L (or a multi-char C/D/M)
        # so single-letter C/D/M cannot match section labels like "C. Permission..."
        # or name initials like "C. F. W. Walther". This regex demands either
        #   (a) starts with I, V, X, or L (optionally followed by more Roman digits), OR
        #   (b) starts with C, D, or M but must be followed by at least one more Roman digit
        # Examples accepted: I., II., IV., IX., XIV., CI., DC., MCM.
        # Examples rejected: C., D., M.  (single letters that could be section/initials)
        r'(?m)^((?:[IVXL][IVXLCDM]*|[CDM][IVXLCDM]+)\.\s+[A-Z].+?)$',
    ]

    articles = []
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                articles.append({
                    "title": match.group(1).strip(),
                    "start": start,
                    "end": end,
                })
            break  # use the first pattern that finds matches

    return articles


# =========================================================================
# CHUNKING
# =========================================================================

def chunk_text(text, chunk_size=1500, overlap_size=200, articles=None):
    """
    Splits text into overlapping chunks.

    Parameters:
        text (str): The full document text
        chunk_size (int): Target size of each chunk in characters.
                          1500 chars is roughly 250 words, which is a good
                          size for translation — enough context for the AI
                          to understand, small enough for detailed evaluation.
        overlap_size (int): How many characters of overlap between chunks.
                            200 chars (~30 words) gives enough context.
        articles (list): Article boundaries from detect_articles().
                         If provided, chunks won't cross article boundaries.

    Returns:
        list: List of chunk dicts with text, context, and metadata
    """
    chunks = []
    chunk_id = 1

    # If we have article structure, chunk within each article separately.
    # This ensures a chunk about "Justification" doesn't bleed into
    # a chunk about "The Church" — the theological context stays clean.
    if articles:
        # PREAMBLE CATCH: anything before the first detected article header
        # (introductions, abstracts, epigraphs, table of contents, opening
        # paragraphs) MUST still be translated. Without this, any text before
        # articles[0]["start"] was being silently dropped — that's how the
        # wauwatosa and otto deliveries ended up starting halfway through.
        if articles[0]["start"] > 0:
            preamble_text = text[:articles[0]["start"]].strip()
            if preamble_text:
                preamble_chunks = _split_into_chunks(
                    preamble_text, chunk_size, overlap_size, chunk_id
                )
                for chunk in preamble_chunks:
                    chunk["article"] = "(preamble)"
                    chunks.append(chunk)
                    chunk_id += 1

        for article in articles:
            article_text = text[article["start"]:article["end"]].strip()
            if not article_text:
                continue

            article_chunks = _split_into_chunks(
                article_text, chunk_size, overlap_size, chunk_id
            )

            for chunk in article_chunks:
                chunk["article"] = article["title"]
                chunks.append(chunk)
                chunk_id += 1
    else:
        # No structure detected — just chunk the whole text
        chunks = _split_into_chunks(text, chunk_size, overlap_size, chunk_id)
        for chunk in chunks:
            chunk["article"] = None

    # Add context_before and context_after (the overlap)
    for i, chunk in enumerate(chunks):
        chunk["context_before"] = chunks[i - 1]["text"][-overlap_size:] if i > 0 else ""
        chunk["context_after"] = chunks[i + 1]["text"][:overlap_size] if i < len(chunks) - 1 else ""
        chunk["position"] = f"{i + 1}/{len(chunks)}"

    # COVERAGE SANITY CHECK: a chunker bug in April 2026 silently dropped
    # everything before the first detected article header, so wauwatosa and
    # otto deliveries started translating halfway through the source. Never
    # again. If the chunks together don't cover at least 90% of the source
    # text, something is wrong — refuse to proceed and make the caller fix it.
    # (We use 90% rather than 100% to allow for whitespace trimming, empty
    # article bodies, and the natural gaps that .strip() removes.)
    chunked_total = sum(len(c["text"]) for c in chunks)
    source_total = len(text.strip())
    if source_total > 0:
        coverage = chunked_total / source_total
        if coverage < 0.90:
            raise ValueError(
                f"Chunker coverage sanity check FAILED: chunks contain only "
                f"{chunked_total:,} chars ({coverage * 100:.1f}%) of source text "
                f"({source_total:,} chars). This usually means detect_articles() "
                f"matched a line that isn't actually a section header and "
                f"silently discarded the content before it. Inspect the source, "
                f"tighten the header patterns, and re-run. Refusing to proceed."
            )

    return chunks


def _split_into_chunks(text, chunk_size, overlap_size, start_id):
    """
    Internal function: splits a block of text into chunks, trying to
    break at sentence boundaries rather than mid-sentence.
    """
    chunks = []
    current_pos = 0
    chunk_id = start_id

    while current_pos < len(text):
        # Take a chunk_size slice
        end_pos = min(current_pos + chunk_size, len(text))

        # If we're not at the end, try to break at a sentence boundary
        if end_pos < len(text):
            # Look backwards from end_pos for a period, question mark, or
            # exclamation mark followed by a space or newline
            search_region = text[current_pos:end_pos]
            last_sentence_end = -1

            for match in re.finditer(r'[.!?]\s', search_region):
                last_sentence_end = match.end()

            if last_sentence_end > chunk_size * 0.5:
                # Found a sentence break in the second half — use it
                end_pos = current_pos + last_sentence_end

        chunk_text_content = text[current_pos:end_pos].strip()

        if chunk_text_content:  # don't add empty chunks
            chunks.append({
                "chunk_id": f"{chunk_id:03d}",
                "text": chunk_text_content,
            })
            chunk_id += 1

        # Move forward, but step back by overlap_size for context
        current_pos = end_pos

    return chunks


# =========================================================================
# FILE HANDLING
# =========================================================================

def ingest_document(filepath, source_language=None):
    """
    Reads a document file and returns the full text.
    Supports .txt and .pdf files.

    Parameters:
        filepath (str): Path to the source document
        source_language (str): Override the source language from config

    Returns:
        dict: {"text": full_text, "language": language, "filename": name}
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".pdf":
        import subprocess
        result = subprocess.run(
            ["pdftotext", filepath, "-"],
            capture_output=True, text=True
        )
        text = result.stdout
    elif ext == ".json":
        # Support Benjamin's chunks.json format
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data and "text" in data[0]:
            text = "\n\n".join(item["text"] for item in data)
        else:
            raise ValueError(f"Unrecognized JSON format in {filepath}")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .txt, .pdf, or .json")

    return {
        "text": text,
        "language": source_language or SOURCE_LANGUAGE,
        "filename": os.path.basename(filepath),
    }


def save_chunks(chunks, output_path):
    """Save chunks to a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to {output_path}")


# =========================================================================
# CLI ENTRY POINT
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Ingest a document and split it into chunks"
    )
    parser.add_argument(
        "input", type=str,
        help="Path to the source document (.txt, .pdf, or .json)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for the chunks JSON file"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1500,
        help="Target chunk size in characters (default: 1500)"
    )
    parser.add_argument(
        "--overlap", type=int, default=200,
        help="Overlap between chunks in characters (default: 200)"
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Source language override (default: from config.py)"
    )
    args = parser.parse_args()

    # Ingest
    print(f"Reading {args.input}...")
    doc = ingest_document(args.input, args.language)
    print(f"  Language: {doc['language']}")
    print(f"  Text length: {len(doc['text'])} characters")

    # Detect structure
    print("Detecting document structure...")
    articles = detect_articles(doc["text"])
    if articles:
        print(f"  Found {len(articles)} articles/sections:")
        for a in articles[:10]:
            print(f"    - {a['title']}")
        if len(articles) > 10:
            print(f"    ... and {len(articles) - 10} more")
    else:
        print("  No article structure detected — chunking as continuous text")

    # Chunk
    print(f"Chunking (size={args.chunk_size}, overlap={args.overlap})...")
    chunks = chunk_text(
        doc["text"],
        chunk_size=args.chunk_size,
        overlap_size=args.overlap,
        articles=articles if articles else None,
    )
    print(f"  Created {len(chunks)} chunks")

    # Save
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "data",
        f"chunks_{os.path.splitext(doc['filename'])[0]}.json"
    )
    save_chunks(chunks, output_path)

    # Summary
    print(f"\nChunk summary:")
    for c in chunks[:3]:
        article = f" [{c['article']}]" if c['article'] else ""
        print(f"  {c['chunk_id']}: {len(c['text'])} chars{article} — {c['text'][:60]}...")
    if len(chunks) > 3:
        print(f"  ... ({len(chunks) - 3} more chunks)")


if __name__ == "__main__":
    main()
