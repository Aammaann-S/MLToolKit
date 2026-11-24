#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTENSION="$ROOT/vscode-extension"
if [[ ! -f "$ROOT/MLToolKit/catalog.json" ]]; then
  python "$ROOT/scripts/build_snippets.py"
fi
(cd "$EXTENSION" && npm install && npm run compile)

install_into() {
  local root="$1"
  if [[ -d "$(dirname "$root")" ]]; then
    local dest="$root/local.mltoolkit-1.1.0"
    rm -rf "$dest"
    mkdir -p "$dest/out"
    cp "$EXTENSION/package.json" "$EXTENSION/catalog.json" "$dest/"
    cp -R "$EXTENSION/out/." "$dest/out/"
    echo "Installed to $dest"
  fi
}

install_into "$HOME/.vscode/extensions"
install_into "$HOME/.cursor/extensions"
echo "Reload the editor. pip does not install this extension."
