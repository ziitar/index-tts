# PROJECT KNOWLEDGE BASE

**Generated:** 2025-05-29
**Commit:** 830f6f8
**Branch:** main

## OVERVIEW

IndexTTS2 is a zero-shot text-to-speech system by Bilibili. Python 3.10+, PyTorch, `uv` package manager (mandatory). Two inference backends: v1 (legacy) and v2 (current with emotion control).

## STRUCTURE

```
.
├── indextts/              # Main package (56K LOC)
│   ├── gpt/               # Autoregressive GPT models (v1 + v2)
│   ├── s2mel/             # Speech-to-mel pipeline (v2 only)
│   ├── BigVGAN/           # Neural vocoder (vendored NVIDIA)
│   ├── accel/             # GPU acceleration (FlashAttention, CUDA graphs)
│   ├── utils/             # Shared utilities + maskgct codecs
│   ├── vqvae/             # VQ-VAE (v1 only)
│   ├── infer.py           # IndexTTS v1 inference
│   ├── infer_v2.py        # IndexTTS2 v2 inference (main)
│   └── cli.py             # CLI entry (v1 only, TODO: v2)
├── webui.py               # Gradio WebUI (primary user-facing)
├── checkpoints/           # Model weights (gitignored except *.yaml)
├── examples/              # Demo audio files
├── tests/                 # Ad-hoc test scripts (no framework)
├── tools/                 # gpu_check.py, i18n/
└── pyproject.toml         # Build config, deps, entry points
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| TTS inference (v2) | `indextts/infer_v2.py` | `IndexTTS2` class, emotion control |
| TTS inference (v1) | `indextts/infer.py` | `IndexTTS` class, legacy |
| Web UI | `webui.py` | Gradio, uses v2 |
| CLI | `indextts/cli.py` | Uses v1 only (TODO: v2) |
| GPT model | `indextts/gpt/model_v2.py` | `UnifiedVoice` v2 |
| Audio pipeline | `indextts/s2mel/` | CFM, DiT, vocoders |
| Text processing | `indextts/utils/front.py` | `TextNormalizer`, `TextTokenizer` |
| Speaker embedding | `indextts/s2mel/modules/campplus/DTDNN.py` | CAMPPlus |
| Emotion control | `indextts/infer_v2.py:719` | `QwenEmotion` class |
| Config | `checkpoints/config.yaml` | OmegaConf, architecture params |

## CONVENTIONS

- **Package manager**: `uv` ONLY. No pip, no conda. Issues from non-uv installs rejected.
- **Build**: `hatchling` backend. `uv sync --all-extras` to install.
- **PyTorch CUDA**: From `https://download.pytorch.org/whl/cu128`, never PyPI.
- **DeepSpeed**: `no-build-isolation-package` to find CUDA PyTorch.
- **Platform deps**: `wetext` on non-Linux, `WeTextProcessing` on Linux.
- **No linter config**: ruff/mypy used locally but not configured in-repo.
- **No test framework**: Ad-hoc scripts in `tests/`, no pytest.

## ANTI-PATTERNS (THIS PROJECT)

- **DO NOT** use pip/conda — only `uv` supported
- **DO NOT** manually activate venvs before `uv run` — conflicts
- **NEVER** commit binary model files — only `checkpoints/*.yaml` tracked
- **NEVER** use `use_cuda_kernel=True` without matching nvcc+ninja
- **DEPRECATED** CLI (`cli.py`) uses v1 — not v2
- **WARNING** CUDA fused kernel incorrect for non-default hyperparams

## UNIQUE STYLES

- **Vendor-heavy**: BigVGAN, DAC, OpenVoice, maskGCT embedded in source
- **Dual inference**: v1 (infer.py) and v2 (infer_v2.py) coexist
- **Emotion modes**: Audio ref, emotion vector, text description (QwenEmotion)
- **Anti-aliased activations**: Snake/SnakeBeta + upsample→act→downsample pattern everywhere

## COMMANDS

```bash
# Setup
uv sync --all-extras

# WebUI
uv run webui.py

# CLI (v1 only)
indextts "text" -v voice.wav -o output.wav

# GPU check
uv run tools/gpu_check.py

# Python API
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints")
tts.infer(spk_audio_prompt='examples/voice_01.wav', text='Hello', output_path="gen.wav")
```

## NOTES

- Repository history was reset — delete local copies and re-clone
- Models auto-download from HuggingFace/ModelScope on first run
- Use `HF_ENDPOINT="https://hf-mirror.com"` if HuggingFace is slow
- FP16 recommended for speed + lower VRAM (small quality loss)
- DeepSpeed may slow inference — test both with/without
