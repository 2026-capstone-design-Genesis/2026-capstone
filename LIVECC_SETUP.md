# LiveCC clip caption application

This is an offline clip QA application of the public model. It does not
reproduce LiveCC streaming training/commentary. Two seconds and Korean
single-sentence output are application experiment conditions.

## Verified environment (2026-09-12)

- Windows, Python 3.12.13.
- Separate model environment: `../.venv-livecc`.
- torch 2.5.1+cu121, torchvision 0.20.1+cu121, transformers 4.52.4,
  accelerate 1.7.0, bitsandbytes 0.46.0, PyAV 12.0.0.
- RTX 3060 12 GiB. A real CUDA `Linear4bit` NF4/BF16 kernel test passed.
- Official model revision: dbe7415c5d2024c3e2d7f898ecbc58fd9fc8da2f.
- The 15 pinned inference files (16,598,765,267 bytes) were downloaded and
  size-verified under `.livecc/model/<revision>`.
- Real-time environment: av==12.0.0 installed for PTS decoding.
- Full NF4 model loading was attempted but did not complete: the first shard
  remained at 0% and its interrupted Windows CUDA process left a stale GPU
  context. Full loading, Korean generation, latency and peak VRAM remain unverified.

Official references:
https://github.com/showlab/livecc/blob/main/inference.md
https://github.com/showlab/livecc
https://huggingface.co/chenjoya/LiveCC-7B-Instruct

## Configuration

Existing OpenAI captions remain the default. In the project .env:

```dotenv
CAPTION_BACKEND=livecc
LIVECC_MODEL_PATH=C:\absolute\path\to\Capstone_CluLLM\.livecc\model\dbe7415c5d2024c3e2d7f898ecbc58fd9fc8da2f
LIVECC_CLIP_MODE=centered
LIVECC_CLIP_SECONDS=2
LIVECC_FRAME_COUNT=4
```

Use 8 frames for the comparison condition, or 1 for the keyframe baseline.
Do not switch the real `.env` until the full model test below succeeds. Restart
the real-time server after changing settings. The selected snapshot
must be local; inference never implicitly downloads weights. For a copied
snapshot, model_revision.txt must contain its verified revision.

The BF16 profile retains the official Qwen2VL, Liger and FlashAttention 2 path
and its 18 GiB preflight guard. The separate RTX 3060 candidate uses NF4,
BF16 compute and SDPA without Liger. It fixes the entire device map to CUDA 0
and rejects CPU/disk model placement; there is no automatic fallback.

## Data contract

livecc_caption.decode_clip reads actual PTS with PyAV in two passes. Incoming
recordings use their verified per-frame .timestamps.json capture-time mapping.
Selection clips the requested window, includes the nearest in-window keyframe,
deduplicates timestamps, sorts observations, and retains fewer frames for short
clips. select_indices(..., ended=False) returns waiting until the requested
future interval is available. Stored finalized recordings use ended=True.

The processor receives exactly the decoded RGB arrays as one video, never a
path that it can resample. Odd observation counts get one explicitly recorded
duplicate for Qwen temporal patch pairing. Nonuniform time intervals are NOT
encoded in positional inputs; the prompt forbids speed/duration claims.
Generation does not reuse previous histories/caches. One format retry is
allowed; original output and error status are retained.

caption_details records settings, times, source indices, PTS, RGB hashes, status
and model revision. Successful inference also records versions, latency and peak
allocated/reserved VRAM. Failed captions do not silently switch to OpenAI. Existing overall report
summarization remains OpenAI-based; no translation model is introduced.
Cross-batch caption merging is disabled in LiveCC mode so a 2-second caption
cannot be silently replaced by a caption of a combined long clip.

## Verification and remaining work

Logic tests:

```powershell
.\.venv-realtime-local\Scripts\python.exe -m unittest discover -s tests -v
```

Pinned NF4 load and 1/4/4/8-frame comparison after a clean GPU restart:

```powershell
..\.venv-livecc\Scripts\python.exe -u test_livecc_model.py --video <recording.webm> --time <session-second>
```

When `CAPTION_BACKEND=livecc`, `run_realtime_app.py` starts one hidden worker.
The OS lock prevents a second web server from loading another model. Durable
SQLite jobs, worker heartbeat, pending/running/waiting UI, and per-batch result
matching survive web-server restarts.

Sampling tests cover boundaries, keyframe inclusion, order, short inputs,
past-only, cross-file waiting, clip PTS, durable queue and worker heartbeat.
No successful full-model inference, Korean quality, hallucination rate,
1/4/8-frame domain comparison or model VRAM/latency benchmark has been completed.
No paper accuracy numbers are application measurements.
The referenced 02_방법론_설계서.md was not found in the workspace.
