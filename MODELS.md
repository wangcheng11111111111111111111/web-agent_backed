# Model Assets

The repository does not contain model weights, user-trained RVC timbres, uploaded
voice prompts, or runtime media. The following public assets must be downloaded
under `backend/models/` or `backend/vendor/RVC/assets/`.

## Download

After installing the Python dependencies:

```powershell
python scripts/download_models.py
```

Useful variants:

```powershell
python scripts/download_models.py --only cosyvoice
python scripts/download_models.py --only rvc
python scripts/download_models.py --source huggingface
```

The script downloads CosyVoice2 from `iic/CosyVoice2-0.5B` on ModelScope, falls
back to `FunAudioLLM/CosyVoice2-0.5B` on Hugging Face, and downloads HuBERT and
RMVPE from `lj1995/VoiceConversionWebUI`.

## Required Layout

```text
backend/
├── models/
│   ├── cosyvoice2/
│   │   ├── cosyvoice2.yaml
│   │   ├── llm.pt
│   │   ├── flow.pt
│   │   ├── hift.pt
│   │   ├── campplus.onnx
│   │   ├── speech_tokenizer_v2.onnx
│   │   └── CosyVoice-BlankEN/
│   └── rvc/
│       ├── weights/
│       │   └── <voice>.pth
│       └── indexes/
│           └── <voice>*.index
└── vendor/RVC/assets/
    ├── hubert_base/
    │   ├── config.json
    │   ├── preprocessor_config.json
    │   └── pytorch_model.bin
    └── rmvpe/
        └── rmvpe.pt
```

User-trained `.pth` and `.index` files are not redistributed. Put them in the
directories shown above; the frontend can select and combine them independently.

## Configuration

The defaults in `.env.example` already point to these paths. For a separate
deployment, configure:

```ini
LOCAL_TTS_ENGINE=cosyvoice
LOCAL_TTS_MODEL_DIR=models/cosyvoice2
LOCAL_TTS_DEVICE=cuda
LOCAL_RVC_MODEL_DIR=models/rvc/weights
LOCAL_RVC_INDEX_DIR=models/rvc/indexes
LOCAL_RVC_DEVICE=cuda
```

CUDA is recommended. Set the device values to `cpu` only when the corresponding
runtime has enough CPU time and memory for interactive speech generation.
