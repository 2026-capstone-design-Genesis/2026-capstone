from pathlib import Path
import ast
import json
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from modules.hf_cache import configure_hf_cache
from modules.model_manager import ModelManager


def _load_prompt_template() -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "importance_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def _safe_parse_score(text: str) -> float:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = ast.literal_eval(match.group(0))
            score = float(obj.get("score", 0))
            return max(0.0, min(10.0, score))
        except Exception:
            pass
    match = re.search(r"score[^0-9]*(\d+(?:\.\d+)?)", text, flags=re.I)
    if match:
        return max(0.0, min(10.0, float(match.group(1))))
    match = re.search(r"\b(10|[0-9])\b", text)
    if match:
        return float(match.group(1))
    return 0.0


class Llama2ImportanceScorer:
    def __init__(self, cfg: dict, hf_token: str | None = None, model=None, tokenizer=None):
        self.cfg = cfg
        self.model_id = cfg["model_id"]
        self.window_size = int(cfg.get("window_size", 7))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 32))
        self.hf_token = hf_token
        self.cache_dir = configure_hf_cache(cfg.get("cache_dir"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token, use_fast=True, cache_dir=self.cache_dir)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quant_cfg = None
            if cfg.get("use_quantization_4bit", True) and self.device == "cuda":
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=self.hf_token,
                cache_dir=self.cache_dir,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                quantization_config=quant_cfg,
                low_cpu_mem_usage=True,
            )
            if self.device != "cuda":
                self.model.to(self.device)
            self.model.eval()
        self.base_instruction = _load_prompt_template()

    def build_query(self, segments: list[dict], center_idx: int) -> tuple[str, int, int]:
        half = self.window_size // 2
        start = max(0, center_idx - half)
        end = min(len(segments), center_idx + half + 1)
        window = segments[start:end]
        central_no = center_idx - start + 1

        lines = [
            f"Please evaluate the importance score of the central segment #{central_no} in the following {len(window)} segments.",
            "",
        ]
        for i, seg in enumerate(window, start=1):
            caption = seg.get("caption") or "No caption."
            lines.append(f"#{i}: {caption}")
        lines.append("")
        lines.append("Return only {\"score\": integer_between_0_and_10}.")
        return "\n".join(lines), start, end

    def build_prompt(self, query: str) -> str:
        return f"<s>[INST] {self.base_instruction}\n\n{query} [/INST]"

    @torch.inference_mode()
    def score_one(self, segments: list[dict], center_idx: int) -> dict:
        query, _, _ = self.build_query(segments, center_idx)
        prompt = self.build_prompt(query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        raw_score = _safe_parse_score(answer)

        result = {
            "segment_id": segments[center_idx]["segment_id"],
            "raw_score_0_10": raw_score,
            "local_score": raw_score / 10.0,
            "llm_answer": answer,
            "query": query,
        }

        if self.cfg.get("extract_embeddings", True):
            emb = self.extract_query_answer_embedding(prompt, answer, query)
            result["embedding"] = emb.astype(np.float32)
        return result

    @torch.inference_mode()
    def extract_query_answer_embedding(self, prompt: str, answer: str, query: str) -> np.ndarray:
        # Rerun full prompt + answer to obtain final hidden states after the Llama RMSNorm path.
        full_text = prompt + " " + answer
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.last_hidden_state[0]  # [seq, dim]

        query_ids = self.tokenizer(query, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        all_ids = inputs["input_ids"][0].detach().cpu().tolist()

        def find_subseq(haystack, needle):
            if not needle:
                return None
            for i in range(0, len(haystack) - len(needle) + 1):
                if haystack[i:i+len(needle)] == needle:
                    return i, i + len(needle)
            return None

        q_span = find_subseq(all_ids, query_ids)
        a_span = find_subseq(all_ids, answer_ids)

        chunks = []
        if q_span:
            chunks.append(hidden[q_span[0]:q_span[1]])
        if a_span:
            chunks.append(hidden[a_span[0]:a_span[1]])
        if not chunks:
            chunks.append(hidden)
        token_emb = torch.cat(chunks, dim=0)

        if self.cfg.get("pooling", "max") == "mean":
            pooled = token_emb.mean(dim=0)
        else:
            pooled = token_emb.max(dim=0).values
        return pooled.detach().float().cpu().numpy()


def score_segments_with_llama2(segments: list[dict], cfg: dict, output_dir: str, hf_token: str | None = None, model_manager: ModelManager | None = None) -> tuple[list[dict], np.ndarray | None]:
    cache_path = Path(output_dir) / "scores_raw.json"
    emb_path = Path(output_dir) / "llm_embeddings.npy"

    if cfg.get("cache", True) and cache_path.exists():
        scores = json.loads(cache_path.read_text(encoding="utf-8"))
        embeddings = np.load(emb_path) if emb_path.exists() else None
        return scores, embeddings

    llama_bundle = model_manager.load_llama(cfg, hf_token=hf_token) if model_manager else None
    scorer = Llama2ImportanceScorer(
        cfg,
        hf_token=hf_token,
        model=llama_bundle.model if llama_bundle else None,
        tokenizer=llama_bundle.tokenizer if llama_bundle else None,
    )
    scores = []
    embeddings = []
    for idx in tqdm(range(len(segments)), desc="Scoring with Llama-2"):
        result = scorer.score_one(segments, idx)
        emb = result.pop("embedding", None)
        if emb is not None:
            embeddings.append(emb)
        scores.append(result)

    cache_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    emb_arr = None
    if embeddings:
        emb_arr = np.stack(embeddings, axis=0)
        np.save(emb_path, emb_arr)
    return scores, emb_arr
