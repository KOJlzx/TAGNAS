#!/usr/bin/env bash
# =========================================================================
# Download the processed Reddit / Amazon-Baby / Citation graphs that are too
# large to ship in git, and unpack them into data/<dataset>/raw/.
#
#   bash data/download_data.sh
#
# The archive holds our processed `raw/{graph,node-data,texts}.csv` for the
# three large graphs (derived from the public TEG-DB benchmark -- see
# data/DOWNLOAD.md for provenance). T-REx / PubMed / Twitter are already in
# the repo and are not part of this archive.
#
# Env:
#   DATA_URL   override the download URL (default: the hosted copy below)
# =========================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

# Anonymized read-only host (Open Science Framework). Direct-download URL of
# the archive; safe to leave as-is.
DATA_URL="${DATA_URL:-https://osf.io/download/2p7tf/?view_only=7fe3c6ffa5f9409ba15c23f0e7f155b9}"
SHA256="2092db7b7d9bceccf3c0cd6fa2c2ee8eb8d83ab214ac12f6c5500d7575bebe6d"
ARCHIVE="tagnas_processed_data.tar.gz"

echo "Downloading processed data archive (~241 MB) ..."
echo "  from: $DATA_URL"
curl -fL --retry 3 -o "$ARCHIVE" "$DATA_URL"

echo "Verifying checksum ..."
echo "${SHA256}  ${ARCHIVE}" | sha256sum -c - || {
    echo "Checksum mismatch -- download corrupted or URL changed." >&2
    exit 1
}

echo "Unpacking into data/<dataset>/raw/ ..."
tar -xzf "$ARCHIVE"
rm -f "$ARCHIVE"

echo
echo "Done. Verify:"
for ds in reddit amazon-baby citation; do
    printf '  data/%s/raw : %s\n' "$ds" "$(ls data/$ds/raw/*.csv 2>/dev/null | wc -l) csv files"
done
echo "Derived caches (data/<ds>/derived/*.pt) are rebuilt automatically on first run."
