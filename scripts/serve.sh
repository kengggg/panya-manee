#!/bin/sh
# Serve the dashboard locally for preview.
# Usage: sh scripts/serve.sh
#   then open http://localhost:8000
set -e
cd "$(dirname "$0")/../site"
echo "Serving site/ at http://localhost:8000"
python3 -m http.server 8000
