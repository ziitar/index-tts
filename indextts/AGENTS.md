# indextts/ — Main Package

## OVERVIEW

Core TTS package. 56K LOC across 7 submodules. Two inference backends: v1 (infer.py) and v2 (infer_v2.py).

## STRUCTURE

```
indextts/
├── infer.py           # IndexTTS v1 (legacy)
├── infer_v2.py        # IndexTTS2 v2 (current, 851 lines)
├── cli.py             # CLI wrapper (v1 only, TODO: v2)
├── gpt/               # GPT autoregressive models
├── s2mel/             # Speech-to-mel pipeline (v2 only)
├── BigVGAN/           # Neural vocoder (vendored)
├── accel/             # GPU acceleration
├── utils/             # Shared utilities + codecs
└── vqvae/             # VQ-VAE (v1 only, mostly unused)
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Main inference (v2) | `infer_v2.py` | `IndexTTS2` class, 851 lines |
| Legacy inference (v1) | `infer.py` | `IndexTTS` class, 690 lines |
| Emotion control | `infer_v2.py:719` | `QwenEmotion` text→emotion |
| CLI entry | `cli.py` | Only v1, has TODO for v2 |
| Config loading | `infer_v2.py:76` | `OmegaConf.load(cfg_path)` |
| Model loading | `infer_v2.py:80-170` | GPT, s2mel, BigVGAN, CAMPPlus |
| Text processing | `utils/front.py` | `TextNormalizer` + `TextTokenizer` |

## CONVENTIONS

- **Relative imports**: `from .gpt.model_v2 import UnifiedVoice`
- **Config access**: `OmegaConf.load()` then attribute access (`cfg.gpt.dim`)
- **Checkpoint loading**: `load_checkpoint()` from `utils/checkpoint.py`
- **Device auto-detection**: cuda → xpu → mps → cpu

## ANTI-PATTERNS

- **DO NOT** import from `infer.py` for new code — use `infer_v2.py`
- **DO NOT** use `cli.py` for IndexTTS2 — it only supports v1
- **NEVER** hardcode device — use auto-detection pattern

## INFERENCE FLOW (v2)

```
Text → TextNormalizer → TextTokenizer (BPE)
  → w2v-bert semantic features + CAMPPlus speaker embedding
  → QwenEmotion (text→emotion vector)
  → GPT v2 (autoregressive codes)
  → s2mel (CFM/DiT diffusion → mel spectrogram)
  → BigVGAN (mel → waveform)
```
