#!/usr/bin/env bash
# Deploy this repo to a Hugging Face Docker Space.
#
# Prereqs: HF_TOKEN (write) exported, and `huggingface_hub` installed
#   (already in the dev env via `uv run`).
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/deploy_hf.sh <hf-username> [space-name]
#
# After the first deploy, set the ANTHROPIC_API_KEY secret in the Space UI
# (Settings -> Variables and secrets) — never commit it.
set -euo pipefail

USER="${1:?usage: deploy_hf.sh <hf-username> [space-name]}"
SPACE="${2:-production-rag-platform}"
REPO_ID="$USER/$SPACE"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

: "${HF_TOKEN:?export HF_TOKEN (a write token from https://hf.co/settings/tokens)}"

echo "==> Creating Space $REPO_ID (Docker SDK) if it doesn't exist"
python - "$REPO_ID" <<'PY'
import os, sys
from huggingface_hub import create_repo
repo_id = sys.argv[1]
create_repo(repo_id, repo_type="space", space_sdk="docker",
            token=os.environ["HF_TOKEN"], exist_ok=True)
print("ok:", repo_id)
PY

echo "==> Uploading repo (with HF frontmatter README, without the local Qdrant index)"
python - "$REPO_ID" "$ROOT" <<'PY'
import os, sys, tempfile, shutil, pathlib
from huggingface_hub import upload_folder

repo_id, root = sys.argv[1], pathlib.Path(sys.argv[2])
staging = pathlib.Path(tempfile.mkdtemp())

ignore = shutil.ignore_patterns(
    ".git", ".venv", "data/qdrant", "*.sqlite3", "*.db",
    ".pytest_cache", ".ruff_cache", "__pycache__", ".env",
)
for item in root.iterdir():
    if item.name in {".git", ".venv"}:
        continue
    dest = staging / item.name
    if item.is_dir():
        shutil.copytree(item, dest, ignore=ignore)
    else:
        shutil.copy2(item, dest)

# HF Spaces needs frontmatter in README.md
shutil.copy2(root / "deploy" / "README_hf.md", staging / "README.md")

upload_folder(
    repo_id=repo_id, repo_type="space", folder_path=str(staging),
    token=os.environ["HF_TOKEN"], commit_message="Deploy Production RAG Platform",
)
shutil.rmtree(staging)
print("uploaded to", repo_id)
PY

echo "==> Done. Space: https://huggingface.co/spaces/$REPO_ID"
echo "    Set the ANTHROPIC_API_KEY secret in the Space settings, then it will (re)build."
