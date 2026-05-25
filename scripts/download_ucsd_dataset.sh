#!/usr/bin/env bash
# Fetch Julian McAuley's UCSD Steam dataset into data/raw/.
#
# Three files are pulled (the 1.3 GB steam_reviews dump is intentionally
# skipped — we get sufficient interaction signal from the australian
# users_items file, and the Australian reviews file already carries
# binary 'recommend' labels):
#
#   steam_games.json.gz             ~2.6 MB   game catalogue + content metadata
#   australian_users_items.json.gz  ~71 MB    per-user owned-game lists + playtime
#   australian_user_reviews.json.gz ~6.6 MB   per-user reviews + recommend labels
#
# All three files are Python repr() format (single quotes, u'...' literals),
# not JSON. The loader in gamereco.datasets.steam_ucsd parses them with
# ast.literal_eval.

set -euo pipefail

DEST_DIR="${1:-data/raw}"
mkdir -p "$DEST_DIR"

declare -a FILES=(
  "https://cseweb.ucsd.edu/~wckang/steam_games.json.gz"
  "https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_users_items.json.gz"
  "https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz"
)

for url in "${FILES[@]}"; do
  name=$(basename "$url")
  out="$DEST_DIR/$name"
  if [ -s "$out" ]; then
    echo "✓ $out already present ($(wc -c < "$out") bytes), skipping"
    continue
  fi
  echo "→ fetching $url"
  curl -sSL --fail -o "$out.tmp" "$url"
  mv "$out.tmp" "$out"
  echo "✓ saved $out ($(wc -c < "$out") bytes)"
done

echo "done"
