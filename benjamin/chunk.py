import json
import nltk
import re
import numpy as np
from sentence_transformers import SentenceTransformer

nltk.download('punkt_tab')

INPUT_TXT = "/home/benjamin/ge_rig/output/corrected/walther_final.txt"
OUTPUT_JSON = "/home/benjamin/ge_rig/output/chunks.json"

BIBLICAL_BOOKS = [
    "Gen", "Ex", "Lev", "Num", "Dtn", "Jos", "Ri", "Rut", "Sam", "Kön",
    "Chr", "Esra", "Neh", "Est", "Hiob", "Ps", "Spr", "Pred", "Hld",
    "Jes", "Jer", "Klgl", "Hes", "Dan", "Hos", "Joel", "Am", "Ob",
    "Jon", "Mi", "Nah", "Hab", "Zef", "Hag", "Sach", "Mal",
    "Mt", "Mk", "Lk", "Joh", "Apg", "Röm", "Kor", "Gal", "Eph",
    "Phil", "Kol", "Thess", "Tim", "Tit", "Phlm", "Hebr", "Jak",
    "Pet", "Jud", "Offb",
    # Latin/common variants Walther uses
    "Matth", "Marc", "Luc", "Act", "Rom", "Cor", "Thim",
]
 
# Other common abbreviations in 19th-century German theological prose
OTHER_ABBREVS = [
    "z", "B", "d", "h", "u", "a", "s", "o", "g", "vgl", "Vgl",
    "Dr", "Prof", "Hr", "St", "Nr", "Jh", "Jhrh", "ca", "etc",
    "bzw", "ggf", "evtl", "bes", "eig", "eigtl",
    "Anm", "Bd", "Sp", "Kap", "Art", "Abs",
    # Archaic
    "rc",   # ꝛc. — the archaic etc.
]
 
def protect_abbreviations(text):
    """Replace periods in known abbreviations with a placeholder."""
    protected = text
 
    # Biblical book references: "Jes. 61, 1" → "Jes§ 61, 1"
    for book in BIBLICAL_BOOKS:
        protected = re.sub(
            rf'\b({re.escape(book)})\.\s*(\d)',
            r'\1§ \2',
            protected
        )
 
    # Other abbreviations: "z. B." → "z§ B§"
    for abbrev in OTHER_ABBREVS:
        protected = re.sub(
            rf'\b({re.escape(abbrev)})\.',
            r'\1§',
            protected
        )
 
    # ꝛc. (archaic rc.) — catch the Unicode variant too
    protected = re.sub(r'ꝛc\.', 'ꝛc§', protected)
 
    return protected
 
def restore_abbreviations(text):
    return text.replace('§', '.')
 
 
# ── Preprocessing ──────────────────────────────────────────────────────────────
 
def preprocess_with_headers(text):
    # Fix hyphenated line breaks (OCR artifact)
    text = re.sub(r'-\n', '', text)
 
    lines = text.split('\n')
 
    segments = []
    current_lecture = None
    current_thesis = None
    current_lines = []
 
    for line in lines:
        header_match = re.match(r'#+\s+(.+)$', line)
        if header_match:
            # Save accumulated text before processing new header
            if current_lines:
                segments.append({
                    "lecture": current_lecture,
                    "thesis": current_thesis,
                    "text": ' '.join(current_lines).strip()
                })
                current_lines = []
 
            header_text = header_match.group(1).strip()
 
            # Classify the header by content
            if re.match(r'Thesis\s+\w+', header_text, re.IGNORECASE):
                current_thesis = header_text
            elif re.match(r'\w+e\s+Abendvorlesung', header_text, re.IGNORECASE):
                current_lecture = header_text
                # New lecture resets thesis context — a thesis must be
                # explicitly re-declared. Change to `pass` if Walther
                # continues a thesis across lectures without re-announcing it.
                current_thesis = None
            else:
                # Unknown header type — store as thesis for now
                current_thesis = header_text
 
        else:
            cleaned = line.strip()
            if cleaned:
                current_lines.append(cleaned)
 
    # Flush final segment
    if current_lines:
        segments.append({
            "lecture": current_lecture,
            "thesis": current_thesis,
            "text": ' '.join(current_lines).strip()
        })
 
    return segments
 
 
# ── Chunking ───────────────────────────────────────────────────────────────────
 
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def split_segment(sentences, embeddings, target_words=400, min_to_split=600, max_words=900):
    total_words = sum(len(s.split()) for s in sentences)

    if total_words < min_to_split or len(sentences) <= 1:
        return [" ".join(sentences)]

    n_chunks = max(2, round(total_words / target_words))

    if len(sentences) <= n_chunks:
        return [" ".join(sentences)]

    similarities = [
        cosine_similarity(embeddings[i], embeddings[i+1])
        for i in range(len(embeddings) - 1)
    ]

    cut_points = set(
        sorted(range(len(similarities)), key=lambda i: similarities[i])[:n_chunks - 1]
    )

    # Split into sentence lists (not joined strings yet)
    segments = []
    current = [0]  # track sentence indices
    for i in range(len(similarities)):
        if i in cut_points:
            segments.append(current)
            current = [i + 1]
        else:
            current.append(i + 1)
    segments.append(current)

    # For each segment, check if it's still too large and recurse if needed
    results = []
    for seg_indices in segments:
        seg_sentences = [sentences[i] for i in seg_indices]
        seg_words = sum(len(s.split()) for s in seg_sentences)

        if seg_words > max_words and len(seg_sentences) > 1:
            seg_embeddings = embeddings[seg_indices[0]:seg_indices[-1] + 1]
            # Recurse — but with min_to_split=0 to force splitting regardless of size
            sub_chunks = split_segment(
                seg_sentences, seg_embeddings,
                target_words=target_words,
                min_to_split=0,
                max_words=max_words
            )
            results.extend(sub_chunks)
        else:
            results.append(" ".join(seg_sentences))

    return results
 
def semantic_chunk(text, model, target_words=800, min_chunk_words=50):
    # Protect abbreviations before sentence tokenization
    protected = protect_abbreviations(text)
    sentences_protected = nltk.sent_tokenize(protected, language='german')
    sentences = [restore_abbreviations(s) for s in sentences_protected]

    # Merge very short sentences before embedding
    merged = []
    buffer = ""
    for s in sentences:
        if buffer:
            s = buffer + " " + s
            buffer = ""
        if len(s.split()) < 10:
            buffer = s
        else:
            merged.append(s)
    if buffer:
        if merged:
            merged[-1] += " " + buffer
        else:
            merged.append(buffer)
    sentences = merged

    if len(sentences) <= 1:
        return [" ".join(sentences)] if sentences else []

    embeddings = model.encode(sentences, show_progress_bar=False)

    print(f"  sentences: {len(sentences)}, words: {sum(len(s.split()) for s in sentences)}, target chunks: {round(sum(len(s.split()) for s in sentences) / target_words)}")

    # Fixed-k splitting based on target word count
    chunks = split_segment(sentences, embeddings, target_words=target_words)

    # Post-process: merge any chunks that are still too short
    result = []
    for chunk in chunks:
        if len(chunk.split()) < min_chunk_words and result:
            result[-1] += " " + chunk
        else:
            result.append(chunk)

    return result
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    model = SentenceTransformer("BAAI/bge-m3")
 
    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        raw = f.read()
 
    print(f"Raw file: {len(raw)} characters")
 
    # Preprocess — splits text into header-tagged segments
    segments = preprocess_with_headers(raw)
    print(f"Segments after header parsing: {len(segments)}")
 
    total_segment_chars = sum(len(s["text"]) for s in segments)
    print(f"Total segment characters: {total_segment_chars}")
 
    # Chunk each segment independently, preserving metadata
    results = []
    for i, segment in enumerate(segments):
        # Normalize whitespace within segment
        text = re.sub(r' +', ' ', segment["text"]).strip()
 
        chunks = semantic_chunk(text, model, target_words=800, min_chunk_words=50)
 
        for chunk in chunks:
            results.append({
                "lecture": segment["lecture"],
                "thesis": segment["thesis"],
                "text": chunk
            })
 
    print(f"Produced {len(results)} chunks")
    total_chunk_chars = sum(len(r["text"]) for r in results)
    print(f"Total chunk characters: {total_chunk_chars}")
    print(f"Coverage: {total_chunk_chars / total_segment_chars * 100:.1f}%")
 
    # Preview first and last few chunks
    print("\n── First chunk ──")
    print(f"  lecture: {results[0]['lecture']}")
    print(f"  thesis:  {results[0]['thesis']}")
    print(f"  text:    {results[0]['text'][:200]}...")
 
    print("\n── Last chunk ──")
    print(f"  lecture: {results[-1]['lecture']}")
    print(f"  thesis:  {results[-1]['thesis']}")
    print(f"  text:    {results[-1]['text'][:200]}...")
 
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
 
    print("\nSaved to chunks.json")