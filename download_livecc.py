"""Download a pinned inference snapshot only after NF4 kernel validation."""
import json
from pathlib import Path
import shutil
import os
os.environ.setdefault('HF_XET_CACHE', str(Path(__file__).parent / '.livecc' / 'xet'))
os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).parent / '.livecc' / 'hub'))
os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES', '0')
from huggingface_hub import HfApi, snapshot_download
from livecc_caption import MODEL_ID, REVISION

root = Path(__file__).parent / '.livecc'
report = json.loads((root / 'kernel-report.json').read_text())
assert report['status'] == 'kernel_ok'
info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
assert info.sha == REVISION
files = [f for f in info.siblings if f.rfilename.endswith(('.safetensors', '.json', '.txt', '.model', '.jinja'))]
size = sum(f.size or 0 for f in files)
free = shutil.disk_usage(root).free
model_dir = root / 'model' / REVISION
completed = sum((model_dir / f.rfilename).stat().st_size for f in files if (model_dir / f.rfilename).is_file())
partial = sum(path.stat().st_size for path in (model_dir / '.cache' / 'huggingface' / 'download').glob('*.incomplete')) \
    if (model_dir / '.cache' / 'huggingface' / 'download').is_dir() else 0
remaining_disk = max(0, size - completed - min(partial, size - completed))
# One streaming copy plus 8 GiB for recordings/system operations.
if free < remaining_disk + 8 * 1024**3:
    raise RuntimeError(f'Disk budget failed: free={free}, remaining={remaining_disk}, reserve=8 GiB')
manifest = {'revision': REVISION, 'download_bytes': size, 'free_before': free,
            'completed_bytes': completed, 'partial_bytes': partial, 'remaining_disk_bytes': remaining_disk,
            'files': [{'name': f.rfilename, 'bytes': f.size} for f in files]}
(root / 'download-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest), flush=True)
path = snapshot_download(MODEL_ID, revision=REVISION, local_dir=model_dir,
                         allow_patterns=[f.rfilename for f in files], max_workers=1)
verified = []
for file in files:
    destination = model_dir / file.rfilename
    if not destination.is_file() or destination.stat().st_size != file.size:
        raise RuntimeError(f'Download verification failed: {file.rfilename}')
    verified.append({'name': file.rfilename, 'bytes': destination.stat().st_size})
(model_dir / 'model_revision.txt').write_text(REVISION, encoding='ascii')
(root / 'download-complete.json').write_text(json.dumps({
    'model_id': MODEL_ID, 'revision': REVISION, 'verified_bytes': sum(item['bytes'] for item in verified),
    'files': verified,
}, indent=2), encoding='utf-8')
print(path, flush=True)
