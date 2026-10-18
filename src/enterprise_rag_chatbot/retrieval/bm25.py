"""Small-corpus BM25 with identifier-preserving tokenization."""

import math
import re
from collections import Counter

from enterprise_rag_chatbot.schemas import Chunk, SearchResult


def tokenize(text: str) -> list[str]:
    words = re.findall(r"\w+(?:[-.]\w+)*", text.casefold())
    return words + [part for word in words if "-" in word for part in word.split("-")]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.terms = []
        self.document_frequency = Counter()
        for chunk in chunks:
            fields = " ".join(str(chunk.metadata.get(key, "")) for key in (
                "document_id", "title", "section_title", "version",
            ))
            terms = Counter(tokenize(fields + "\n" + chunk.text))
            self.terms.append(terms)
            self.document_frequency.update(terms.keys())
        self.lengths = [sum(terms.values()) for terms in self.terms]
        self.average_length = sum(self.lengths) / len(chunks) if chunks else 1

    def search(self, query: str, top_k: int, allowed_document_ids: set[str] | None = None) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        results = []
        for chunk, terms, length in zip(self.chunks, self.terms, self.lengths, strict=True):
            if allowed_document_ids is not None and chunk.metadata["document_id"] not in allowed_document_ids:
                continue
            score = 0.0
            for token in set(tokenize(query)):
                frequency = terms[token]
                if not frequency:
                    continue
                df = self.document_frequency[token]
                idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
                norm = self.k1 * (1 - self.b + self.b * length / (self.average_length or 1))
                score += idf * frequency * (self.k1 + 1) / (frequency + norm)
            if score > 0:
                results.append(SearchResult(chunk, score))
        return sorted(results, key=lambda item: (-item.score, item.chunk.id))[:top_k]
