from pathlib import Path
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from modules.model_manager import ModelManager

from modules.hf_cache import configure_hf_cache


class LlavaCaptionGenerator:
    def __init__(self, model_id: str, prompt: str, max_new_tokens: int = 64, hf_token: str | None = None, cache_dir: str | None = None, model=None, processor=None):
        self.model_id = model_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.hf_token = hf_token
        self.cache_dir = configure_hf_cache(cache_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
        else:
            self.processor = AutoProcessor.from_pretrained(model_id, token=self.hf_token, cache_dir=self.cache_dir)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                token=self.hf_token,
                cache_dir=self.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device != "cuda":
                self.model.to(self.device)
            self.model.eval()

    @torch.inference_mode()
    def caption_image(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=self.prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        # Common LLaVA prompt format includes ASSISTANT:. Keep the answer tail.
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1]
        return text.strip()


def generate_segment_captions(segments: list[dict], cfg: dict, output_dir: str, hf_token: str | None = None, model_manager: ModelManager | None = None) -> list[dict]:
    cache_path = Path(output_dir) / "captions.json"
    if cfg.get("cache", True) and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        by_id = {item["segment_id"]: item["caption"] for item in cached}
        for seg in segments:
            seg["caption"] = by_id.get(seg["segment_id"])
        return segments

    if not cfg.get("enabled", True):
        for seg in segments:
            seg["caption"] = ""
        return segments

    backend = cfg.get("backend", "llava")
    if backend != "llava":
        raise ValueError(f"Unsupported caption backend in this MVP: {backend}")

    llava_bundle = model_manager.load_llava(cfg, hf_token=hf_token) if model_manager else None
    generator = LlavaCaptionGenerator(
        model_id=cfg["model_id"],
        prompt=cfg["prompt"],
        max_new_tokens=int(cfg.get("max_new_tokens", 64)),
        hf_token=hf_token,
        cache_dir=cfg.get("cache_dir"),
        model=llava_bundle.model if llava_bundle else None,
        processor=llava_bundle.processor if llava_bundle else None,
    )

    for seg in tqdm(segments, desc="Generating LLaVA captions"):
        seg["caption"] = generator.caption_image(seg["representative_frame_path"])

    cache_path.write_text(json.dumps([
        {"segment_id": s["segment_id"], "caption": s["caption"], "representative_frame_path": s["representative_frame_path"]}
        for s in segments
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    return segments
