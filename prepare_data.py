import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional


TOKEN_RE = re.compile(r"[A-Za-z']+")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")
POSITIVE_WORDS = {"good", "great", "nice", "love", "awesome", "perfect", "glad", "happy", "thanks", "yay"}
NEGATIVE_WORDS = {"bad", "sad", "angry", "hate", "upset", "annoyed", "worried", "stress", "stressed", "sorry"}


@dataclass
class Message:
    speaker: str
    text: str
    timestamp: Optional[str] = None


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def sentiment_score(text: str) -> float:
    toks = tokenize(text)
    if not toks:
        return 0.0
    pos = sum(1 for t in toks if t in POSITIVE_WORDS)
    neg = sum(1 for t in toks if t in NEGATIVE_WORDS)
    return (pos - neg) / max(len(toks), 1)


def load_jsonl(path: str | Path) -> list[Message]:
    p = Path(path)
    rows: list[Message] = []
    # Use utf-8-sig to transparently handle BOM-prefixed exports.
    with p.open("r", encoding="utf-8-sig", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON line {idx}: {exc}") from exc
            speaker = str(obj.get("speaker", "")).strip()
            text = str(obj.get("text", "")).strip()
            if not speaker or not text:
                continue
            rows.append(Message(speaker=speaker, text=text, timestamp=obj.get("timestamp")))
    if not rows:
        raise ValueError("No usable messages were found in input.")
    return rows


def build_pairs(messages: list[Message], friend_name: str) -> list[tuple[str, str]]:
    friend = friend_name.strip().lower()
    pairs: list[tuple[str, str]] = []

    for idx, msg in enumerate(messages):
        if msg.speaker.strip().lower() != friend:
            continue
        prev_user_text = ""
        for j in range(idx - 1, -1, -1):
            prev = messages[j]
            if prev.speaker.strip().lower() != friend and prev.text.strip():
                prev_user_text = prev.text.strip()
                break
        if prev_user_text:
            pairs.append((prev_user_text, msg.text.strip()))
    return pairs


def build_profile(messages: list[Message], friend_name: str) -> dict:
    friend = friend_name.strip().lower()
    replies = [m for m in messages if m.speaker.strip().lower() == friend and m.text.strip()]
    if not replies:
        raise ValueError(f"No messages found for friend '{friend_name}'.")

    lengths = [len(r.text) for r in replies]
    token_counts = [len(tokenize(r.text)) for r in replies]
    sentiments = [sentiment_score(r.text) for r in replies]

    question_rate = sum(1 for r in replies if "?" in r.text) / len(replies)
    exclamation_rate = sum(r.text.count("!") for r in replies) / max(sum(lengths), 1)
    emoji_rate = sum(len(EMOJI_RE.findall(r.text)) for r in replies) / max(sum(lengths), 1)

    lowercase_chars = sum(1 for r in replies for ch in r.text if ch.isalpha() and ch.islower())
    alpha_chars = sum(1 for r in replies for ch in r.text if ch.isalpha())
    lowercase_ratio = lowercase_chars / max(alpha_chars, 1)

    endings = []
    for r in replies:
        parts = [p.strip().lower() for p in re.split(r"[.!?]", r.text) if p.strip()]
        if parts:
            endings.append(parts[-1])

    vocab = Counter(tok for r in replies for tok in tokenize(r.text))
    sentiment_mean = mean(sentiments)

    return {
        "friend_name": friend_name,
        "message_count": len(replies),
        "avg_chars": round(mean(lengths), 2),
        "avg_tokens": round(mean(token_counts), 2),
        "question_rate": round(question_rate, 4),
        "exclamation_density": round(exclamation_rate, 4),
        "emoji_density": round(emoji_rate, 4),
        "lowercase_ratio": round(lowercase_ratio, 4),
        "sentiment_center": round(sentiment_mean, 4),
        "sentiment_variability": round(math.sqrt(mean([(s - sentiment_mean) ** 2 for s in sentiments])), 4),
        "common_endings": [e for e, _ in Counter(endings).most_common(8)],
        "top_words": [w for w, _ in vocab.most_common(20)],
        "interpretation": {
            "tone": "supportive and reassuring" if sentiment_mean >= 0.05 else "neutral-pragmatic",
            "energy": "detailed" if mean(lengths) > 70 else "concise",
            "style": "casual, text-native" if lowercase_ratio > 0.9 else "mixed formality",
        },
    }


def build_style_system_prompt(profile: dict) -> str:
    friend_name = profile.get("friend_name", "friend")
    tone = profile.get("interpretation", {}).get("tone", "neutral-pragmatic")
    energy = profile.get("interpretation", {}).get("energy", "concise")
    style = profile.get("interpretation", {}).get("style", "casual")
    avg_chars = profile.get("avg_chars", 60)
    top_words = profile.get("top_words", [])[:12]
    endings = profile.get("common_endings", [])[:5]

    return (
        f"You are writing as {friend_name}. Mimic their texting style naturally.\n"
        f"- Tone: {tone}\n"
        f"- Energy: {energy}\n"
        f"- Style: {style}\n"
        f"- Typical response length: about {avg_chars} characters\n"
        f"- Frequent words: {', '.join(top_words) if top_words else 'none'}\n"
        f"- Common endings: {', '.join(endings) if endings else 'none'}\n"
        "Stay in character. Do not mention these instructions."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare profile + SFT JSONL for Qwen3 LoRA")
    parser.add_argument("--input", required=True, help="Path to chat JSONL")
    parser.add_argument("--friend", required=True, help="Friend speaker name")
    parser.add_argument("--out", default="data/sft_train.jsonl", help="Output training JSONL")
    parser.add_argument("--out-profile", default="data/profile.json", help="Output profile JSON")
    parser.add_argument("--out-pairs", default="data/pairs.json", help="Output pairs JSON")
    args = parser.parse_args()

    messages = load_jsonl(args.input)
    profile = build_profile(messages, args.friend)
    pairs = build_pairs(messages, args.friend)
    if not pairs:
        raise ValueError("No user->friend training pairs found.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_profile).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_pairs).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_profile, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    with open(args.out_pairs, "w", encoding="utf-8") as f:
        json.dump([{"user_text": u, "friend_reply": a} for u, a in pairs], f, indent=2)

    system_prompt = build_style_system_prompt(profile)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for user_text, friend_reply in pairs:
            sample = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": friend_reply},
                ]
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    print(f"Profile saved: {args.out_profile}")
    print(f"Pairs saved: {args.out_pairs} ({len(pairs)} pairs)")
    print(f"SFT dataset saved: {args.out} ({written} rows)")


if __name__ == "__main__":
    main()
