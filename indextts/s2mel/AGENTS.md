# indextts/s2mel/ — Speech-to-Mel Pipeline

## OVERVIEW

Converts semantic tokens to mel-spectrograms. Contains vocoders (BigVGAN, Vocos, HiFiGAN), diffusion models (CFM/DiT), codec (DAC), and feature extractors. Used only by IndexTTS2 (v2).

## STRUCTURE

```
s2mel/
├── hf_utils.py                 # HuggingFace download helpers
├── optimizers.py               # MultiOptimizer, scheduler defs
├── wav2vecbert_extract.py      # Wav2Vec2-BERT semantic extraction
├── dac/                        # Descript Audio Codec (vendored)
│   ├── model/                  # Encoder→RVQ→Decoder
│   └── nn/                     # Quantize, layers, losses
└── modules/                    # Core neural network modules
    ├── commons.py              # MyModel, build_model(), load_checkpoint2
    ├── flow_matching.py        # CFM (Conditional Flow Matching)
    ├── diffusion_transformer.py # DiT (Diffusion Transformer)
    ├── length_regulator.py     # InterpolateRegulator (duration control)
    ├── quantize.py             # FAquantizer (3-stream quantization)
    ├── audio.py                # Mel-spectrogram extraction
    ├── wavenet.py              # WaveNet blocks
    ├── rmvpe.py                # F0 pitch extraction
    ├── bigvgan/                # NVIDIA BigVGAN vocoder
    ├── vocos/                  # Vocos Fourier-based vocoder
    ├── hifigan/                # HiFi-GAN/HiFiTNet vocoder
    ├── campplus/               # CAMPPlus speaker embedding
    ├── openvoice/              # OpenVoice integration
    ├── gpt_fast/               # Fast Transformer (llama2.c derived)
    └── alias_free_torch/       # Anti-aliased activations
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Model orchestration | `modules/commons.py` | `MyModel`, `build_model()` |
| Mel generation | `modules/flow_matching.py` | `CFM` class, Euler solver |
| Diffusion backbone | `modules/diffusion_transformer.py` | `DiT` class |
| Duration control | `modules/length_regulator.py` | `InterpolateRegulator` |
| Speaker embedding | `modules/campplus/DTDNN.py` | `CAMPPlus` |
| F0 extraction | `modules/rmvpe.py` | `RMVPE` class |
| Vocoder (primary) | `modules/bigvgan/bigvgan.py` | `BigVGAN` class |
| Vocoder (fast) | `modules/vocos/pretrained.py` | `Vocos` class |
| Codec | `dac/model/dac.py` | Encoder→RVQ→Decoder |
| Feature extractor | `wav2vecbert_extract.py` | Wav2Vec2-BERT features |

## CONVENTIONS

- **No __init__.py**: `modules/` has no package init — import by full path
- **Vendor-heavy**: bigvgan/, vocos/, openvoice/, dac/ are adapted external code
- **Anti-aliased**: Snake/SnakeBeta + upsample→act→downsample everywhere
- **Weight norm**: `weight_norm(nn.Conv1d(...))` is standard pattern
- **Mixed origins**: NVIDIA, Meta, Amphion, OpenVoice, SpeechBrain

## KEY CLASSES

| Class | File | Role |
|-------|------|------|
| `MyModel` | `modules/commons.py` | Top-level model container |
| `CFM` | `modules/flow_matching.py` | Conditional Flow Matching |
| `DiT` | `modules/diffusion_transformer.py` | Diffusion Transformer |
| `InterpolateRegulator` | `modules/length_regulator.py` | Duration control |
| `BigVGAN` | `modules/bigvgan/bigvgan.py` | Neural vocoder |
| `CAMPPlus` | `modules/campplus/DTDNN.py` | Speaker embedding |
| `RMVPE` | `modules/rmvpe.py` | Pitch extraction |
| `FAquantizer` | `modules/quantize.py` | 3-stream quantization |

## TRAINING STAGES

1. **DiT stage**: CFM + DiT + InterpolateRegulator → mel generation
2. **Codec stage**: Encoder + FAquantizer → discrete speech codes
3. **Mel_vocos stage**: Vocos decoder → waveform from mel

## ANTI-PATTERNS

- **DO NOT** modify vendored code (bigvgan/, vocos/, openvoice/) unless fixing bugs
- **NEVER** use CUDA kernel BigVGAN without matching nvcc+ninja
- **WARNING** Duplicate `alias_free_torch/` at two locations — use `modules/` one
