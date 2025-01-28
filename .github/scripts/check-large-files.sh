#!/usr/bin/env bash
LIMIT=524288
found_files=$(find . -path ./.git -prune -o -type f -size +${LIMIT}c -print)
if [ -n "$found_files" ]; then
  echo "ERROR: Found files larger than our configured limit of $(( $LIMIT / 1024 )) kB:"
  echo "$found_files"
  exit 1
fi