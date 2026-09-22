#!/usr/bin/env bash
# Build the ggml oracle used by the op-level validation tests.
#
# It links the PINNED ggml from the reference checkout, so what it computes is
# what llama.cpp computes. Rebuild it after changing REFERENCE.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

[ -d llama.cpp ] || { echo "  no reference checkout — run scripts/clone-reference.sh"; exit 1; }

if [ ! -f llama.cpp/build/bin/libggml.so ]; then
    echo "  building ggml (CPU only — the oracle judges numbers, not speed)"
    cmake -B llama.cpp/build -S llama.cpp \
        -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF >/dev/null
    cmake --build llama.cpp/build --target llama-tokenize -j "$(nproc)" >/dev/null
fi

gcc -O2 -o tools/ggml-oracle/oracle tools/ggml-oracle/oracle.c \
    -I llama.cpp/ggml/include \
    -L llama.cpp/build/bin -lggml -lggml-base -lggml-cpu -lm \
    -Wl,-rpath,'$ORIGIN/../../llama.cpp/build/bin'

echo "  built tools/ggml-oracle/oracle"
