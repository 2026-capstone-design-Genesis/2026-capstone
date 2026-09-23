from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from transformers import LlavaForConditionalGeneration

from modules.hf_cache import configure_hf_cache


@dataclass(frozen=True)
class LlavaBundle:
    model: Any
    processor: Any
    model_id: str
    cache_dir: str


@dataclass(frozen=True)
class LlamaBundle:
    model: Any
    tokenizer: Any
    model_id: str
    cache_dir: str
    use_quantization_4bit: bool


class ModelManager:
    """Process-level model cache for LLaVA and Llama.

    Instantiate this once when the CLI/Web UI starts, call ``preload_for_config``
    before processing videos, then pass the same manager into the pipeline.
    The model objects remain in RAM/VRAM while the Python process is alive.
    """

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._llava: dict[tuple[str, str], LlavaBundle] = {}
        self._llama: dict[tuple[str, str, bool], LlamaBundle] = {}

    def _llava_key(self, cfg: dict) -> tuple:
        """
        LLaVA 모델 캐시 key 생성.
        같은 model_id, cache_dir, 4bit 설정이면 같은 모델로 판단.
        """
        return (
            cfg.get("model_id"),
            configure_hf_cache(cfg.get("cache_dir")),
            bool(cfg.get("use_quantization_4bit", True)),
        )


    def _llama_key(self, cfg: dict) -> tuple:
        """
        Llama 모델 캐시 key 생성.
        같은 model_id, cache_dir, 4bit 설정이면 같은 모델로 판단.
        """
        return (
            cfg.get("model_id"),
            configure_hf_cache(cfg.get("cache_dir")),
            bool(cfg.get("use_quantization_4bit", True)),
        )

    def load_llava(self, cfg: dict, hf_token: str | None = None) -> LlavaBundle:
        model_id = cfg["model_id"]
        cache_dir = configure_hf_cache(cfg.get("cache_dir"))
        key = (model_id, cache_dir)
        if key in self._llava:
            return self._llava[key]

        print(f"[ModelManager] Loading LLaVA once: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            model.to(self.device)
        model.eval()

        bundle = LlavaBundle(model=model, processor=processor, model_id=model_id, cache_dir=cache_dir)
        self._llava[key] = bundle
        return bundle

    def load_llama(self, cfg: dict, hf_token: str | None = None) -> LlamaBundle:
        model_id = cfg["model_id"]
        cache_dir = configure_hf_cache(cfg.get("cache_dir"))
        use_4bit = bool(cfg.get("use_quantization_4bit", True))
        key = (model_id, cache_dir, use_4bit)
        if key in self._llama:
            return self._llama[key]

        print(f"[ModelManager] Loading Llama once: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_cfg = None
        if use_4bit and self.device == "cuda":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir=cache_dir,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            quantization_config=quant_cfg,
            low_cpu_mem_usage=True,
        )
        if self.device != "cuda":
            model.to(self.device)
        model.eval()

        bundle = LlamaBundle(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            cache_dir=cache_dir,
            use_quantization_4bit=use_4bit,
        )
        self._llama[key] = bundle
        return bundle

    def preload_for_config(self, cfg: dict, hf_token: str | None = None) -> None:
        """Load the models required by the selected pipeline configuration."""
        caption_cfg = cfg.get("caption", {}) or {}
        scoring_cfg = cfg.get("scoring", {}) or {}

        if caption_cfg.get("enabled", True) and caption_cfg.get("backend", "llava") == "llava":
            self.load_llava(caption_cfg, hf_token=hf_token)

        backend = scoring_cfg.get("backend", "llama2")
        # llama2 mode directly scores with Llama. llmvs_checkpoint mode also uses
        # Llama first to generate embeddings for the checkpoint scorer.
        if backend in {"llama2", "llmvs_checkpoint"}:
            self.load_llama(scoring_cfg, hf_token=hf_token)

    def clear(self) -> None:
        self._llava.clear()
        self._llama.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def has_llava(self, cfg: dict) -> bool:
        """요청한 LLaVA 모델이 이미 RAM/VRAM에 올라와 있는지 확인"""
        return self._llava_key(cfg) in self._llava

    def has_llama(self, cfg: dict) -> bool:
        """요청한 Llama 모델이 이미 RAM/VRAM에 올라와 있는지 확인"""
        return self._llama_key(cfg) in self._llama

    def is_ready_for_config(self, cfg: dict) -> bool:
        """
        현재 config에서 필요한 모델들이 이미 로딩되어 있는지 확인.

        여기서는 UI mode 이름을 보지 않음.
        즉, LLMVS 모드에서 preload한 모델을
        CluFrame + LLMVS 모드에서도 그대로 재사용할 수 있음.
        """
        caption_cfg = cfg.get("caption", {}) or {}
        scoring_cfg = cfg.get("scoring", {}) or {}

        # LLaVA caption model 필요 여부 확인
        if caption_cfg.get("enabled", True) and caption_cfg.get("backend", "llava") == "llava":
            if not self.has_llava(caption_cfg):
                return False

        # Llama scoring model 필요 여부 확인
        backend = scoring_cfg.get("backend", "llama2")
        if backend in {"llama2", "llmvs_checkpoint"}:
            if not self.has_llama(scoring_cfg):
                return False

        return True

    def preload_for_config(self, cfg: dict, hf_token: str | None = None) -> dict:
        """
        현재 config에서 필요한 모델들을 preload.

        이미 로딩된 모델이면 다시 로딩하지 않고 재사용함.
        반환값:
        {
            "llava": "loaded" | "reused" | "not_required",
            "llama": "loaded" | "reused" | "not_required"
        }
        """
        caption_cfg = cfg.get("caption", {}) or {}
        scoring_cfg = cfg.get("scoring", {}) or {}

        status = {
            "llava": "not_required",
            "llama": "not_required",
        }

        if caption_cfg.get("enabled", True) and caption_cfg.get("backend", "llava") == "llava":
            status["llava"] = "reused" if self.has_llava(caption_cfg) else "loaded"
            self.load_llava(caption_cfg, hf_token=hf_token)

        backend = scoring_cfg.get("backend", "llama2")
        if backend in {"llama2", "llmvs_checkpoint"}:
            status["llama"] = "reused" if self.has_llama(scoring_cfg) else "loaded"
            self.load_llama(scoring_cfg, hf_token=hf_token)

        self.last_preload_status = status
        return status

_GLOBAL_MODEL_MANAGER: ModelManager | None = None


def get_global_model_manager() -> ModelManager:
    """
    Streamlit이 아닌 main.py 같은 일반 실행에서도
    같은 Python 프로세스 안에서는 ModelManager를 공유하기 위한 함수.
    """
    global _GLOBAL_MODEL_MANAGER

    if _GLOBAL_MODEL_MANAGER is None:
        _GLOBAL_MODEL_MANAGER = ModelManager()

    return _GLOBAL_MODEL_MANAGER