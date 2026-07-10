#!/usr/bin/env bash
# Sync the local project to the Vast.ai instance.
# Usage: ./scripts/deploy.sh
set -euo pipefail

SSH_HOST="ssh4.vast.ai"
SSH_PORT="36820"
REMOTE_DIR="~/finetune-experiment"

rsync -avz --progress \
  -e "ssh -p ${SSH_PORT}" \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude "outputs" \
  --exclude ".env" \
  ./ "root@${SSH_HOST}:${REMOTE_DIR}/"

echo "Synced. SSH in with:"
echo "  ssh root@${SSH_HOST} -p ${SSH_PORT}"
