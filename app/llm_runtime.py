import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .retrieval import top_examples_scored


class LLMRuntime:
    def __init__(self) -> None:
        self._loaded_model_key: tuple[str, str] | None = None
        self._tokenizer = None
        self._model = None

    def _load(self, model_id: str, adapter_path: str | None) -> None:
        key = (model_id, adapter_path or "")
        if self._loaded_model_key == key and self._model is not None and self._tokenizer is not None:
            return

        offload_dir = Path("app_state") / "offload"
        offload_dir.mkdir(parents=True, exist_ok=True)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            quantization_config=quant_config,
            offload_folder=str(offload_dir),
            low_cpu_mem_usage=True,
        )
        if adapter_path:
            try:
                model = PeftModel.from_pretrained(
                    model,
                    adapter_path,
                    device_map="auto",
                    offload_folder=str(offload_dir),
                )
            except Exception:
                # Fallback to base model so chat remains available even if adapter attach fails.
                pass

        self._model = model
        self._loaded_model_key = key

    @staticmethod
    def _build_system_prompt(profile: dict, examples: list[dict]) -> str:
        friend_name = profile.get("friend_name", "friend")
        tone = profile.get("interpretation", {}).get("tone", "neutral-pragmatic")
        energy = profile.get("interpretation", {}).get("energy", "concise")
        style = profile.get("interpretation", {}).get("style", "casual")
        avg_chars = profile.get("avg_chars", 60)
        top_words = profile.get("top_words", [])[:10]

        base = (
            f"You are {friend_name} in a private text conversation.\n"
            f"Tone: {tone}\n"
            f"Energy: {energy}\n"
            f"Style: {style}\n"
            f"Typical length: around {avg_chars} characters.\n"
            f"Frequent words: {', '.join(top_words) if top_words else 'none'}\n"
            "Keep replies natural and concise.\n"
        )

        if not examples:
            return base

        formatted = []
        for idx, ex in enumerate(examples, start=1):
            formatted.append(
                f"Example {idx}:\nUser: {ex.get('user_text','')}\nAssistant: {ex.get('friend_reply','')}"
            )
        return base + "\nStyle examples:\n" + "\n\n".join(formatted)

    def generate_reply(
        self,
        model_id: str,
        profile_path: str | Path,
        pairs_path: str | Path,
        message: str,
        history: list[dict] | None = None,
        adapter_path: str | None = None,
        retrieval_only: bool = False,
        retrieval_k: int = 4,
        max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_p: float = 0.85,
        top_k: int = 20,
    ) -> str:
        self._load(model_id, adapter_path)
        assert self._tokenizer is not None
        assert self._model is not None

        profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
        scored = top_examples_scored(message, pairs, k=max(1, min(int(retrieval_k), 8)))
        examples = [row for _, row in scored]

        if retrieval_only:
            if not scored:
                return "I do not have a close example for that yet."
            best_score, best = scored[0]
            # If confidence is low, still return the nearest real response to preserve style.
            reply = str(best.get("friend_reply", "")).strip()
            if reply:
                return reply
            if best_score <= 0.0:
                return "I do not have a close example for that yet."
            return str(best.get("user_text", "")).strip() or "I do not have a close example for that yet."

        messages = [{"role": "system", "content": self._build_system_prompt(profile, examples)}]
        if history:
            messages.extend(history[-8:])
        messages.append({"role": "user", "content": message})

        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        reply_ids = output_ids[0][inputs.input_ids.shape[1] :]
        return self._tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


runtime = LLMRuntime()
