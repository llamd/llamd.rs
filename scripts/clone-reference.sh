#!/usr/bin/env bash
# Restore the pinned llama.cpp reference checkout.
#
# The reference is not committed to this repository: it is 614 MB of someone
# else's history, and vendoring it would confuse provenance and make LLAMD's
# own log unreadable. This script puts it back at the exact revision the
# validation harness was last run against.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# shellcheck disable=SC2046
eval $(grep -E '^(url|commit|short)=' REFERENCE)
DEST=llama.cpp

if [ -d "$DEST/.git" ]; then
    cur="$(git -C "$DEST" rev-parse HEAD)"
    if [ "$cur" = "$commit" ]; then
        echo "  reference already at $short"
        exit 0
    fi
    echo "  reference is at ${cur:0:9}, pinned is $short — checking out"
    git -C "$DEST" fetch --quiet origin "$commit" 2>/dev/null || git -C "$DEST" fetch --quiet origin
    git -C "$DEST" checkout --quiet "$commit"
else
    echo "  cloning $url (this is ~614 MB)"
    git clone --quiet "$url" "$DEST"
    git -C "$DEST" checkout --quiet "$commit"
fi
echo "  reference at $(git -C "$DEST" rev-parse --short HEAD)"
