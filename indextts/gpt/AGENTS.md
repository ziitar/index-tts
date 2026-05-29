# indextts/gpt/ — GPT Autoregressive Models

## OVERVIEW

Core autoregressive language model for TTS. Two versions: v1 (model.py) and v2 (model_v2.py). HuggingFace-forked transformer infrastructure (~12K lines in transformers_*.py).

## STRUCTURE

```
gpt/
├── model.py                    # UnifiedVoice v1 (713 lines)
├── model_v2.py                 # UnifiedVoice v2 (796 lines, emotion+speed)
├── transformers_gpt2.py        # GPT-2 implementation (1878 lines)
├── transformers_generation_utils.py  # GenerationMixin (4747 lines)
├── transformers_modeling_utils.py    # PreTrainedModel (5525 lines)
├── transformers_beam_search.py       # Beam search (1013 lines)
├── conformer_encoder.py        # Conformer audio encoder (520 lines)
├── perceiver.py                # Perceiver resampler (277 lines)
└── conformer/                  # Conformer subcomponents
    ├── attention.py            # Multi-headed + relative position attention
    ├── embedding.py            # Positional encodings
    └── subsampling.py          # Linear/Conv2D subsampling
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Main GPT model (v2) | `model_v2.py` | `UnifiedVoice` with emotion/speed |
| GPT model (v1) | `model.py` | `UnifiedVoice` legacy |
| Inference wrapper | `model_v2.py:45` | `GPT2InferenceModel` |
| Conditioning | `model_v2.py:215` | `ConditioningEncoder` |
| Emotion merge | `model_v2.py:700+` | `merge_emovec()` |
| Audio encoder | `conformer_encoder.py` | `ConformerEncoder` |
| Latent compression | `perceiver.py` | `PerceiverResampler` (32 tokens) |
| Attention variants | `transformers_gpt2.py` | vanilla, Flash2, SDPA |

## CONVENTIONS

- **HuggingFace fork**: `transformers_*.py` are customized HuggingFace code — DO NOT modify unless necessary
- **Dual-head**: Separate `text_head` and `mel_head` linear projections
- **Conditioning types**: perceiver, conformer_perceiver, conformer_encoder, gst, default
- **Special tokens**: start_text=0, stop_text=1, start_mel=8192, stop_mel=8193

## KEY CLASSES

| Class | File | Role |
|-------|------|------|
| `UnifiedVoice` | `model_v2.py` | Main GPT TTS model |
| `GPT2InferenceModel` | `model_v2.py` | HF generate() wrapper |
| `GPT2Model` | `transformers_gpt2.py` | Transformer backbone |
| `ConditioningEncoder` | `model_v2.py` | Audio→latent encoder |
| `ConformerEncoder` | `conformer_encoder.py` | Conformer speech encoder |
| `PerceiverResampler` | `perceiver.py` | Variable→fixed length |

## ANTI-PATTERNS

- **DO NOT** modify `transformers_*.py` unless fixing HuggingFace bugs
- **NEVER** call `forward()` directly for inference — use `inference_speech()`
- **WARNING** 20+ `TODO(joao)` deprecation markers in transformers files

## v1 vs v2 DIFFERENCES

| Feature | v1 | v2 |
|---------|----|----|
| Conditioning dim | 100 (mel) | 1024 (latent) |
| Emotion control | ❌ | ✅ |
| Speed/duration | ❌ | ✅ |
| Accel engine | ❌ | ✅ (optional) |
| Forward returns | (loss_text, loss_mel, logits) | mel_logits only |
