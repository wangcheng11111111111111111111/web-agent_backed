#!/usr/bin/env sh
set -eu

BACKEND_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ROOT_DIR=$(CDPATH= cd -- "$BACKEND_DIR/.." && pwd)
FRONTEND_DIR="$ROOT_DIR/frontend"
ENV_FILE="$BACKEND_DIR/.env"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install Docker Engine and the NVIDIA Container Toolkit." >&2
  exit 1
fi

if [ ! -f "$FRONTEND_DIR/Dockerfile" ]; then
  echo "Missing sibling frontend repository: $FRONTEND_DIR" >&2
  echo "Clone web-agent_fronted as the frontend directory next to this backend checkout." >&2
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  cp "$BACKEND_DIR/.env.example" "$ENV_FILE"
  echo "Created .env. Fill in LLM_API_KEY, PG_PASSWORD and EMBEDDING_API_KEY, then run again." >&2
  exit 1
fi

for file in \
  "$BACKEND_DIR/models/cosyvoice2/cosyvoice2.yaml" \
  "$BACKEND_DIR/models/cosyvoice2/llm.pt" \
  "$BACKEND_DIR/models/cosyvoice2/flow.pt" \
  "$BACKEND_DIR/models/cosyvoice2/hift.pt" \
  "$BACKEND_DIR/models/cosyvoice2/campplus.onnx" \
  "$BACKEND_DIR/models/cosyvoice2/speech_tokenizer_v2.onnx" \
  "$BACKEND_DIR/models/cosyvoice2/CosyVoice-BlankEN/config.json" \
  "$BACKEND_DIR/vendor/RVC/assets/hubert_base/config.json" \
  "$BACKEND_DIR/vendor/RVC/assets/hubert_base/pytorch_model.bin" \
  "$BACKEND_DIR/vendor/RVC/assets/rmvpe/rmvpe.pt"
do
  if [ ! -f "$file" ]; then
    echo "Missing model file: $file. See DOCKER.md." >&2
    exit 1
  fi
done

cd "$BACKEND_DIR"
docker compose --env-file "$ENV_FILE" up --build -d
docker compose --env-file "$ENV_FILE" ps

echo
echo "Nebula is starting at http://127.0.0.1:${WEB_PORT:-8080}"
