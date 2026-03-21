import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_system_prompt(profile: dict) -> str:
    friend_name = profile.get("friend_name", "friend")
    tone = profile.get("interpretation", {}).get("tone", "neutral-pragmatic")
    energy = profile.get("interpretation", {}).get("energy", "concise")
    style = profile.get("interpretation", {}).get("style", "casual")
    avg_chars = profile.get("avg_chars", 60)
    top_words = profile.get("top_words", [])[:12]

    return (
        f"You are {friend_name} in a private text conversation.\n"
        f"Tone: {tone}\n"
        f"Energy: {energy}\n"
        f"Style: {style}\n"
        f"Typical length: around {avg_chars} characters.\n"
        f"Frequent words: {', '.join(top_words) if top_words else 'none'}\n"
        "Stay natural and in-character."
    )


def apply_chat_template(tokenizer: AutoTokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Qwen3 base model + optional LoRA adapter")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", default="", help="Optional LoRA adapter directory")
    parser.add_argument("--profile", default="data/profile.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    with open(args.profile, "r", encoding="utf-8") as f:
        profile = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    friend_name = profile.get("friend_name", "friend")
    history = [{"role": "system", "content": build_system_prompt(profile)}]

    print("Ghost Chat (Qwen3-8B). Type /exit to quit.")
    while True:
        user_text = input("you> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            break

        history.append({"role": "user", "content": user_text})
        text = apply_chat_template(tokenizer, history)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        reply_ids = output_ids[0][inputs.input_ids.shape[1] :]
        reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
        print(f"{friend_name}> {reply}")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
