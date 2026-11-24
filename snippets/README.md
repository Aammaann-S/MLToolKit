# Snippets

`scripts/catalog_*.py` is the only snippet source. `python scripts/build_snippets.py` writes the same `catalog.json` into `MLToolKit/` and `vscode-extension/`.

Triggers are published with underscores, for example `ml_loadcsv`. After `import MLToolKit`, type `MLToolKit.ml_loadcsv`.

The trigger list is in `snippet_index.md`. Install the extension with `scripts/install_vscode_extension.ps1` or `scripts/install_vscode_extension.sh`.

Every body is standalone Python. Placeholders such as `${1:X_train}` are filled in by the editor after you accept the suggestion.
