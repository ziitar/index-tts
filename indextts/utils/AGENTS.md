# indextts/utils/ — Shared Utilities

## OVERVIEW

Cross-cutting utilities: text processing, checkpoint loading, audio features, attention blocks. Also contains embedded maskgct codec implementations (deep nesting).

## STRUCTURE

```
utils/
├── front.py            # TextNormalizer + TextTokenizer (718 lines)
├── checkpoint.py       # load_checkpoint()
├── common.py           # load_audio, tokenize_by_CJK_char, make_pad_mask, safe_log
├── feature_extractors.py # FeatureExtractor base + MelSpectrogramFeatures
├── arch_util.py        # AttentionBlock, GroupNorm32, normalization()
├── typical_sampling.py # TypicalLogitsWarper
├── xtransformers.py    # Full x-transformers library (1247 lines)
├── text_utils.py       # Chinese text utilities
├── webui_utils.py      # Gradio helpers
├── maskgct_utils.py    # MaskGCT model builders
└── maskgct/            # Embedded codec implementations (11 levels deep!)
    └── models/codec/   # Amphion, FACodec, SpeechTokenizer, kmeans, VEVO
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Text normalization | `front.py` | `TextNormalizer`, pinyin, glossary |
| Tokenization | `front.py` | `TextTokenizer` (SentencePiece BPE) |
| Checkpoint loading | `checkpoint.py` | `load_checkpoint(model, path)` |
| Audio loading | `common.py` | `load_audio(path, sr)` |
| CJK tokenization | `common.py` | `tokenize_by_CJK_char()` |
| Mel features | `feature_extractors.py` | `MelSpectrogramFeatures` |
| Attention blocks | `arch_util.py` | `AttentionBlock` (used by GPT) |
| Sampling | `typical_sampling.py` | `TypicalLogitsWarper` |
| Codec builders | `maskgct_utils.py` | `build_semantic_model/codec()` |

## CONVENTIONS

- **Direct imports**: `from indextts.utils.front import TextNormalizer`
- **No __init__.py exports**: Each submodule imported independently
- **Checkpoint format**: `torch.load()` + `model.load_state_dict()`
- **CJK handling**: `tokenize_by_CJK_char()` inserts spaces between CJK chars

## SHARED FUNCTIONS (widely used)

| Function | File | Used By |
|----------|------|---------|
| `load_checkpoint` | `checkpoint.py` | `infer.py`, `infer_v2.py` |
| `TextNormalizer` | `front.py` | `infer.py`, `infer_v2.py` |
| `TextTokenizer` | `front.py` | `infer.py`, `infer_v2.py` |
| `tokenize_by_CJK_char` | `common.py` | `front.py` |
| `make_pad_mask` | `common.py` | `gpt/conformer_encoder.py` |
| `AttentionBlock` | `arch_util.py` | `gpt/model.py`, `gpt/model_v2.py` |
| `TypicalLogitsWarper` | `typical_sampling.py` | `gpt/model.py`, `gpt/model_v2.py` |

## ANTI-PATTERNS

- **DO NOT** import from `utils.py` (stale duplicate of `common.py`)
- **DO NOT** modify `xtransformers.py` unless fixing third-party bugs
- **NEVER** hardcode checkpoint paths — use `cfg` from OmegaConf
- **WARNING** `maskgct/` is 11 levels deep — contains full model implementations, not utilities
