#!/bin/sh
# Copy the package into docs/ so GitHub Pages can serve it to Pyodide.
set -e
cd "$(dirname "$0")/.."
rm -rf docs/csat_forge
mkdir -p docs/csat_forge
for f in __init__ problem_engine meta_killer_engine diagnosis numverify plain_format; do
  cp csat_forge/$f.py docs/csat_forge/$f.py
done
