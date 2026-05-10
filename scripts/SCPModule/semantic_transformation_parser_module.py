import json
import logging
import re
from typing import Any, Dict, Optional

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert at understanding image editing instructions.
Convert each instruction into EXACTLY one JSON object with these keys:
- "original concept"
- "edited concept"
- "action type"

Allowed values for "action type" are only:
1) "add"
2) "remove"
3) "attribute change"
4) "replace"
5) "global"

Return strict JSON only, no markdown, no extra text.
"""


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def normalize_action_type(v: Optional[str]) -> str:
    raw = (v or "global").strip().lower()
    if raw == "attribute":
        return "attribute change"
    if raw in {"add", "remove", "replace", "attribute change", "global"}:
        return raw
    return "global"


def normalize_result(parsed: Optional[Dict[str, Any]], raw_output: str) -> Dict[str, Any]:
    if parsed is None:
        return {
            "original concept": None,
            "edited concept": None,
            "action type": "global",
            "_raw_output": raw_output[:500],
        }
    return {
        "original concept": parsed.get("original concept"),
        "edited concept": parsed.get("edited concept"),
        "action type": normalize_action_type(parsed.get("action type")),
    }


class SemanticTransformationParser:
    """Semantic Transformation Parser module."""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        max_memory: Optional[Dict[Any, str]] = None,
    ):
        model_path = self._resolve_model_path(model_path)
        dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        load_kw: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": dtype,
        }
        if max_memory is not None:
            load_kw["max_memory"] = max_memory
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        self.supports_thinking_toggle = self._check_thinking_toggle()
        logger.info("SemanticTransformationParser initialized.")

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        """
        Allow passing HuggingFace cache repo root:
        .../models--xxx--yyy (auto resolve to snapshots/<rev>)
        """
        p = Path(str(model_path)).expanduser()
        if not p.exists() or not p.is_dir():
            return str(model_path)
        snap_dir = p / "snapshots"
        if not snap_dir.is_dir():
            return str(p)
        refs_main = p / "refs" / "main"
        if refs_main.is_file():
            rev = refs_main.read_text(encoding="utf-8").strip()
            cand = snap_dir / rev
            if cand.is_dir():
                return str(cand)
        candidates = sorted(x for x in snap_dir.iterdir() if x.is_dir())
        if candidates:
            return str(candidates[0])
        return str(p)

    def _check_thinking_toggle(self) -> bool:
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return True
        except Exception:
            return False

    @torch.no_grad()
    def parse(self, instruction: str, max_new_tokens: int = 160) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f'Instruction: "{instruction}"\nReturn one-line JSON with keys '
                '"original concept","edited concept","action type".',
            },
        ]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if self.supports_thinking_toggle:
            kwargs["enable_thinking"] = False
        prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.input_device)
        out_ids = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out_ids[0][inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        parsed = extract_json(raw)
        return normalize_result(parsed, raw)
