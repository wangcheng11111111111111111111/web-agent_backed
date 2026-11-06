# Third-Party Notices

This repository vendors portions of the following open-source projects:

## CosyVoice

- Source: <https://github.com/FunAudioLLM/CosyVoice>
- Vendored path: `vendor/CosyVoice`
- License: Apache License 2.0 (`vendor/CosyVoice/LICENSE`)
- Local changes: reduced runtime source tree and adjustments for this
  application's inference pipeline.

## Retrieval-based Voice Conversion WebUI

- Source: <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI>
- Vendored path: `vendor/RVC`
- License: MIT (`vendor/RVC/LICENSE`)
- Local changes: reduced runtime source tree and integration changes.
- Large model assets under `vendor/RVC/assets/` are not committed.

## Model assets

CosyVoice2, HuBERT, RMVPE, RVC user weights, and indexes are not part of this
repository. Refer to `README.md` for the expected directory layout and download
sources.

## Matcha-TTS

- Source: <https://github.com/shivammehta25/Matcha-TTS>
- Vendored path: `vendor/CosyVoice/third_party/Matcha-TTS`
- License: MIT (`vendor/CosyVoice/third_party/Matcha-TTS/LICENSE`)
