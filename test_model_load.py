# test_model_load.py
import time, traceback, os
from transformers import AutoProcessor, logging
from transformers import LlavaForConditionalGeneration

logging.set_verbosity_debug()
token = os.environ.get("HF_TOKEN")
print("processor load start")
t0 = time.time()
try:
    p = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", token=token, use_fast=False)
    print("processor load done", "elapsed:", time.time()-t0)
except Exception as e:
    print("processor EX:", type(e).__name__, e)
    traceback.print_exc()
print("model load start")
t1 = time.time()
try:
    # 주의: 모델 로드는 메모리/디스크/시간 많이 사용합니다.
    m = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        token=token,
        low_cpu_mem_usage=True,
        device_map="auto" if __import__("torch").cuda.is_available() else None,
    )
    print("model load done", "elapsed:", time.time()-t1)
except Exception as e:
    print("model EX:", type(e).__name__, e)
    traceback.print_exc()
print("total elapsed:", time.time()-t0)