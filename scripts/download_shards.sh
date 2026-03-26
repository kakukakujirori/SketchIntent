#!/usr/bin/env bash
set -euo pipefail

# Download SketchGraphs shards 001..128 sequentially.
# Usage:
#   ./download_shards.sh [OUTPUT_DIR]
# If OUTPUT_DIR is omitted, files are saved in the current directory.

BASE_URL="https://sketchgraphs.cs.princeton.edu/shards"
OUT_DIR="${1:-.}"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

for i in $(seq -w 001 128); do
  file="shard_${i}_of_128.tar.zst"
  url="${BASE_URL}/${file}"
  echo "==> Downloading ${file}"
  # -c: continue partial downloads, --tries: retry a few times, --timeout: per-try timeout
  wget -c --tries=5 --timeout=30 "$url"
 done

echo "All requested shards processed."