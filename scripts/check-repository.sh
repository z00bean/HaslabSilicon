#!/usr/bin/env sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

set -eu

required_paths='README.md LICENSE NOTICE docs hardware/rtl hardware/testbenches simulation fpga software compiler runtime onnx reference benchmarks scripts .github/workflows'
for path in $required_paths; do
  if [ ! -e "$path" ]; then
    printf '%s\n' "Missing required path: $path" >&2
    exit 1
  fi
done

if ! grep -q '^/haslab-site/$' .gitignore; then
  printf '%s\n' 'haslab-site must remain ignored.' >&2
  exit 1
fi

printf '%s\n' 'Repository foundation checks passed. No implementation targets were run.'
