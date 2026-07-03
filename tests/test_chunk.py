from rag.ingest.chunk import chunk_article, estimate_tokens, split_sections

SAMPLE = """Machine learning is a field of study in artificial intelligence. \
It develops algorithms that learn from data. These models generalize to unseen data.

== History ==
The term machine learning was coined in 1959 by Arthur Samuel. \
Early work focused on pattern recognition. Research continued through several decades.

=== Early days ===
Perceptrons were an early model. They were studied extensively.

== See also ==
List of datasets.

== References ==
Some citation text here.
"""

ARTICLE = {
    "title": "Machine learning",
    "url": "https://en.wikipedia.org/wiki/Machine_learning",
    "text": SAMPLE,
}


def test_split_sections_labels_intro_and_headings():
    sections = split_sections(SAMPLE)
    names = [name for name, _ in sections]
    assert names[0] == "Introduction"
    assert "History" in names


def test_split_sections_drops_boilerplate():
    names = [name for name, _ in split_sections(SAMPLE)]
    assert "See also" not in names
    assert "References" not in names


def test_subsection_text_kept_under_parent_section():
    sections = dict(split_sections(SAMPLE))
    assert "Perceptrons were an early model." in sections["History"]


def test_chunks_have_metadata_and_stable_ids():
    chunks = chunk_article(ARTICLE, min_tokens=5)
    assert chunks, "expected at least one chunk"
    assert all(c.title == "Machine learning" for c in chunks)
    assert all(c.url.endswith("/Machine_learning") for c in chunks)
    # ids are deterministic across runs (uuid5 of title|section|index)
    again = chunk_article(ARTICLE, min_tokens=5)
    assert [c.id for c in chunks] == [c.id for c in again]


def test_long_section_is_split_with_overlap():
    long_text = " ".join(f"Sentence number {i} contains some words." for i in range(200))
    article = {"title": "T", "url": "u", "text": long_text}
    chunks = chunk_article(article, max_tokens=100, overlap_tokens=20, min_tokens=5)
    assert len(chunks) > 1
    assert all(estimate_tokens(c.text) <= 130 for c in chunks)  # budget + one sentence slack
    # consecutive chunks share overlapping sentences
    assert chunks[0].text.split()[-4:] == chunks[1].text.split()[:4] or any(
        s in chunks[1].text for s in chunks[0].text.split(". ")[-2:]
    )


def test_tiny_sections_skipped():
    article = {"title": "T", "url": "u", "text": "Short.\n\n== Stub ==\nTiny."}
    assert chunk_article(article) == []
