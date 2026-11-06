from __future__ import annotations

import argparse
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
COSYVOICE_DIR = BACKEND_DIR / "models" / "cosyvoice2"
RVC_ASSETS_DIR = BACKEND_DIR / "vendor" / "RVC" / "assets"

MODELSCOPE_COSYVOICE2 = "iic/CosyVoice2-0.5B"
HUGGINGFACE_COSYVOICE2 = "FunAudioLLM/CosyVoice2-0.5B"
HUGGINGFACE_RVC_ASSETS = "lj1995/VoiceConversionWebUI"

REQUIRED_FILES = (
    COSYVOICE_DIR / "cosyvoice2.yaml",
    COSYVOICE_DIR / "llm.pt",
    COSYVOICE_DIR / "flow.pt",
    COSYVOICE_DIR / "hift.pt",
    COSYVOICE_DIR / "campplus.onnx",
    COSYVOICE_DIR / "speech_tokenizer_v2.onnx",
    COSYVOICE_DIR / "CosyVoice-BlankEN" / "config.json",
    RVC_ASSETS_DIR / "hubert_base" / "config.json",
    RVC_ASSETS_DIR / "hubert_base" / "pytorch_model.bin",
    RVC_ASSETS_DIR / "rmvpe" / "rmvpe.pt",
)


def _download_cosyvoice(source: str) -> None:
    COSYVOICE_DIR.mkdir(parents=True, exist_ok=True)

    if source in {"auto", "modelscope"}:
        try:
            from modelscope import snapshot_download

            print(f"Downloading CosyVoice2 from ModelScope: {MODELSCOPE_COSYVOICE2}")
            snapshot_download(
                MODELSCOPE_COSYVOICE2,
                local_dir=str(COSYVOICE_DIR),
            )
            return
        except Exception as exc:
            if source == "modelscope":
                raise RuntimeError(f"ModelScope download failed: {exc}") from exc
            print(f"ModelScope unavailable, falling back to Hugging Face: {exc}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing download dependency. Install with "
            "`python -m pip install -U huggingface_hub modelscope`."
        ) from exc

    print(f"Downloading CosyVoice2 from Hugging Face: {HUGGINGFACE_COSYVOICE2}")
    snapshot_download(
        repo_id=HUGGINGFACE_COSYVOICE2,
        local_dir=str(COSYVOICE_DIR),
    )


def _download_rvc_assets(source: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing download dependency. Install with "
            "`python -m pip install -U huggingface_hub`."
        ) from exc

    RVC_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading RVC assets from Hugging Face: {HUGGINGFACE_RVC_ASSETS}")
    snapshot_download(
        repo_id=HUGGINGFACE_RVC_ASSETS,
        local_dir=str(RVC_ASSETS_DIR),
        allow_patterns=["hubert_base/*", "rmvpe/*"],
    )


def _report_missing() -> bool:
    missing = [path for path in REQUIRED_FILES if not path.is_file()]
    if not missing:
        print("All required CosyVoice2, HuBERT and RMVPE files are ready.")
        return False

    print("Missing required files:", file=sys.stderr)
    for path in missing:
        print(f"  - {path.relative_to(BACKEND_DIR)}", file=sys.stderr)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the public model assets required by CosyVoice2 + RVC."
    )
    parser.add_argument(
        "--only",
        choices=("all", "cosyvoice", "rvc"),
        default="all",
        help="Download only one model group.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "modelscope", "huggingface"),
        default="auto",
        help="CosyVoice2 source. RVC assets always use Hugging Face.",
    )
    args = parser.parse_args()

    try:
        if args.only in {"all", "cosyvoice"}:
            _download_cosyvoice(args.source)
        if args.only in {"all", "rvc"}:
            _download_rvc_assets(args.source)
    except Exception as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1

    return 1 if _report_missing() else 0


if __name__ == "__main__":
    raise SystemExit(main())
