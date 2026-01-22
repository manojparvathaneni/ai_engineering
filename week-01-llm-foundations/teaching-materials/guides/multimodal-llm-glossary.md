# Multimodal LLM Architecture

## Diagram
See: `multimodal-llm-architecture.mermaid`

---

## Glossary

### Encoders
Convert raw input into a format the LLM can understand.

| Term | Description |
|------|-------------|
| **Vision Encoder** | Converts images into numerical representations (embeddings). Examples: CLIP, ViT |
| **Audio Encoder** | Converts sound waves into embeddings. Examples: Whisper, wav2vec |
| **Video Encoder** | Processes sequences of frames + audio into embeddings |
| **Tokenizer** | Splits text into tokens (subwords/words) and maps them to numerical IDs |

### Core Components

| Term | Description |
|------|-------------|
| **Tokens** | The basic units the LLM operates on — can represent words, subwords, or encoded representations of other modalities |
| **Embeddings** | Dense numerical vectors that capture meaning; tokens get converted to embeddings before processing |
| **LLM** | Large Language Model — the "reasoning core" that processes token sequences and generates output tokens |

### Decoders
Convert LLM output back into human-usable formats.

| Term | Description |
|------|-------------|
| **Image Decoder** | Generates images from LLM output tokens. Often a diffusion model |
| **Audio Decoder** | Converts audio tokens to waveforms. Examples: vocoders, neural audio synthesis |
| **Video Decoder** | Generates video frames from token sequences |
| **Detokenizer** | Maps token IDs back to text — simple lookup, not a neural network |

---

## Key Insight

The LLM itself only "speaks" tokens. Encoders and decoders are **translators** that let it communicate in other modalities. This modular design allows components to be upgraded or swapped independently.
