"""Run actual CUDA NF4 math before downloading the model."""
import json
import importlib.metadata
from pathlib import Path
import shutil
import torch
import bitsandbytes as bnb

def main():
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 compute unsupported')
    x = torch.randn(4, 128, device='cuda', dtype=torch.bfloat16)
    layer = bnb.nn.Linear4bit(128, 64, bias=False, compute_dtype=torch.bfloat16,
        compress_statistics=True, quant_type='nf4').to('cuda')
    with torch.no_grad():
        y = layer(x)
    torch.cuda.synchronize()
    assert y.shape == (4, 64) and torch.isfinite(y).all()
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    report = {'status': 'kernel_ok', 'gpu': torch.cuda.get_device_name(),
        'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'av')},
        'free_disk_bytes': shutil.disk_usage('.').free, 'free_vram_bytes': torch.cuda.mem_get_info()[0]}
    folder = Path(__file__).parent / '.livecc'
    folder.mkdir(exist_ok=True)
    (folder / 'kernel-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
