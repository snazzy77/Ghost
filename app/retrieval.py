import json
import math
import re
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def cosine_counts(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_pairs(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def top_examples(query: str, pairs: list[dict], k: int = 4) -> list[dict]:
    return [row for _, row in top_examples_scored(query, pairs, k=k)]


def top_examples_scored(query: str, pairs: list[dict], k: int = 4) -> list[tuple[float, dict]]:
    qv = Counter(tokenize(query))
    scored: list[tuple[float, dict]] = []
    for row in pairs:
        user_text = str(row.get("user_text", "")).strip()
        if not user_text:
            continue
        score = cosine_counts(qv, Counter(tokenize(user_text)))
        if query.lower() in user_text.lower():
            score += 0.05
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]
