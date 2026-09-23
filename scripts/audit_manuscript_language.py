#!/usr/bin/env python3
"""Report reproducible language-pattern statistics for the English manuscript."""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

from docx import Document


WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def words(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit_manuscript_language.py MANUSCRIPT.docx")

    path = Path(sys.argv[1])
    document = Document(path)
    body_paragraphs: list[str] = []
    abstract = ""
    in_references = False
    expect_abstract = False

    for paragraph in document.paragraphs:
        text = " ".join(paragraph.text.split())
        if not text:
            continue
        if text == "References":
            in_references = True
            continue
        if in_references:
            continue
        if text == "Abstract":
            expect_abstract = True
            continue
        if expect_abstract:
            abstract = text
            body_paragraphs.append(text)
            expect_abstract = False
            continue
        if text.startswith("Abstract "):
            abstract = text.removeprefix("Abstract ").strip()
            body_paragraphs.append(abstract)
            continue
        if paragraph.style and paragraph.style.name.startswith("Heading"):
            continue
        if text.startswith(("Table ", "Fig. ", "Keywords ")):
            continue
        body_paragraphs.append(text)

    prose = " ".join(body_paragraphs)
    sentences = [s.strip() for s in SENTENCE_RE.split(prose) if len(words(s)) >= 4]
    tokens = words(prose)

    normalized_sentences = collections.Counter(
        re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()
        for s in sentences
    )
    repeated_sentences = [
        (sentence, count)
        for sentence, count in normalized_sentences.items()
        if count > 1
    ]

    eight_grams = collections.Counter(
        tuple(tokens[index : index + 8]) for index in range(max(0, len(tokens) - 7))
    )
    repeated_eight_grams = [
        (" ".join(gram), count)
        for gram, count in eight_grams.items()
        if count > 1
    ]

    starters = collections.Counter(
        " ".join(words(sentence)[:3]) for sentence in sentences if len(words(sentence)) >= 3
    )
    common_starters = [(starter, count) for starter, count in starters.most_common(15) if count >= 3]

    moving_ttr_values = []
    window = 100
    for index in range(0, max(0, len(tokens) - window + 1), window):
        sample = tokens[index : index + window]
        moving_ttr_values.append(len(set(sample)) / len(sample))
    moving_ttr = sum(moving_ttr_values) / len(moving_ttr_values) if moving_ttr_values else 0.0

    sentence_lengths = [len(words(sentence)) for sentence in sentences]
    mean_sentence_length = (
        sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
    )
    sentence_length_sd = (
        (sum((length - mean_sentence_length) ** 2 for length in sentence_lengths)
         / len(sentence_lengths)) ** 0.5
        if sentence_lengths
        else 0.0
    )

    print(f"file={path}")
    print(f"abstract_words={len(words(abstract))}")
    print(f"prose_words={len(tokens)}")
    print(f"sentences={len(sentences)}")
    print(f"mean_sentence_words={mean_sentence_length:.2f}")
    print(f"sentence_length_sd={sentence_length_sd:.2f}")
    print(f"moving_ttr_100={moving_ttr:.3f}")
    print(f"exact_repeated_sentences={len(repeated_sentences)}")
    for sentence, count in repeated_sentences[:10]:
        print(f"  sentence x{count}: {sentence[:180]}")
    print(f"repeated_8grams={len(repeated_eight_grams)}")
    for phrase, count in sorted(repeated_eight_grams, key=lambda item: (-item[1], item[0]))[:15]:
        print(f"  8gram x{count}: {phrase}")
    print("common_3word_sentence_starters=")
    for starter, count in common_starters:
        print(f"  {starter}: {count}")


if __name__ == "__main__":
    main()
