#!/usr/bin/env bash
# Fetch the public Steam-200k dataset (Tamber's Kaggle dump, 200K real
# user-game interactions, ~8.5 MB) into data/raw/steam-200k.csv.
#
# This is the dataset every benchmark in benchmarks/ runs against.

set -euo pipefail

DEST="${1:-data/raw/steam-200k.csv}"
mkdir -p "$(dirname "$DEST")"

PRIMARY="https://raw.githubusercontent.com/warchildmd/game2vec/master/data/steam-200k.csv"
FALLBACKS=(
  "https://raw.githubusercontent.com/lix229/steamer/master/data/steam-200k.csv"
  "https://raw.githubusercontent.com/ragoragino/kaggle-steam/master/data/steam-200k.csv"
  "https://raw.githubusercontent.com/alexretana/Steam-Kaggle-Exploratory-Data-Analysis/master/steam-200k.csv"
)

try() {
  local url="$1"
  echo "→ trying $url"
  if curl -sSL --fail -o "$DEST.tmp" "$url"; then
    mv "$DEST.tmp" "$DEST"
    echo "✓ saved $DEST ($(wc -l < "$DEST") rows)"
    return 0
  fi
  rm -f "$DEST.tmp"
  return 1
}

if try "$PRIMARY"; then
  exit 0
fi
for url in "${FALLBACKS[@]}"; do
  if try "$url"; then exit 0; fi
done

echo "✗ could not fetch dataset from any mirror" >&2
exit 1
