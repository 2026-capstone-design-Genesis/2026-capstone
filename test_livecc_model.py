"""Explicit pinned NF4 load + repeated independent clip test."""
import argparse
import faulthandler
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import psutil
import torch
from livecc_caption import LiveCCCaptioner, REVISION

parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True)
parser.add_argument('--time', type=float, required=True)
args = parser.parse_args()
faulthandler.dump_traceback_later(120, repeat=True)
root = Path(__file__).parent / '.livecc'
model = LiveCCCaptioner(root / 'model' / REVISION, profile='nf4')
started = time.perf_counter()
print('Loading pinned NF4 model', flush=True)
environment = {
    'python': platform.python_version(), 'platform': platform.platform(),
    'gpu': torch.cuda.get_device_name(0), 'vram_before_load_bytes': torch.cuda.mem_get_info()[0],
    'versions': {name: importlib.metadata.version(name) for name in
        ('torch', 'torchvision', 'transformers', 'accelerate', 'bitsandbytes', 'av', 'psutil')},
}
try:
    model.load()
except Exception as error:
    report = {'stage': 'load', 'status': type(error).__name__, 'error': str(error),
              'seconds': time.perf_counter() - started, 'environment': environment}
    (root / 'inference-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    raise
load_seconds = time.perf_counter() - started
environment['vram_after_load_bytes'] = torch.cuda.mem_get_info()[0]
environment['process_ram_after_load_bytes'] = psutil.Process().memory_info().rss
modules = {name: {'class': type(module).__name__, 'dtype': str(module.weight.dtype), 'device': str(module.weight.device)}
           for name, module in model.model.named_modules() if hasattr(module, 'weight') and module.weight is not None}
(root / 'module-placement.json').write_text(json.dumps(modules, indent=2), encoding='utf-8')
results = []
for run, count in enumerate((1, 4, 4, 8), start=1):
    print(f'Inference frames={count}', flush=True)
    result = model.caption(args.video, args.time, count=count, seconds=2, mode='centered')
    result['comparison_run'] = run
    result['one_frame_baseline'] = 'repeated-frame video padding' if count == 1 else None
    result['process_ram_bytes'] = psutil.Process().memory_info().rss
    results.append(result)
    (root / 'inference-report.json').write_text(json.dumps({
        'model_revision': REVISION, 'profile': 'nf4-bf16-sdpa-no-offload',
        'load_seconds': load_seconds, 'environment': environment, 'results': results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True), flush=True)
