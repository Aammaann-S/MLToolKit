# MLToolKit

MLToolKit is an installable Python package and a VS Code extension. The package holds reusable machine-learning implementations. The extension notices `import MLToolKit` and, when you type `MLToolKit.ml_`, inserts the actual Python for that operation.

Selecting CSV loading inserts code like this:

```python
import pandas as pd

df = pd.read_csv("data.csv")
```

It does not insert `MLToolKit.load_csv()`. The inserted code runs without the package. Pip does not install the editor extension. Each piece is installed once per environment.

## Install the package

From a clone:

```powershell
python -m pip install -e .
```

From GitHub, after this repository is published:

```bash
pip install git+https://github.com/USERNAME/MLToolKit.git
```

Optional libraries stay out of the default install:

```powershell
python -m pip install -e ".[boosting,deploy]"
```

Those extras are also listed in `requirements-optional.txt`.

```python
import MLToolKit

MLToolKit.__version__
MLToolKit.load_catalog()
```

Importing `MLToolKit` does not import pandas or scikit-learn. Numbered modules under `MLToolKit/implementations` are loaded when a workflow or test asks for them.

## Install the extension

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_vscode_extension.ps1
```

Or build a VSIX:

```powershell
cd vscode-extension
npm install
npx vsce package --allow-missing-repository
code --install-extension mltoolkit-1.1.0.vsix
```

Reload the editor. In a Python file:

```python
import MLToolKit

MLToolKit.ml_
```

| You type | Suggestions include |
| --- | --- |
| `MLToolKit.ml_load` | `ml_loadcsv`, `ml_loaddataset`, `ml_loadexcel`, `ml_loadjson` |
| `MLToolKit.ml_scale` | `ml_standardscale`, `ml_minmaxscale`, `ml_robustscale`, `ml_maxabsscale`, `ml_normalize` |
| `MLToolKit.ml_impute` | `ml_imputemean`, `ml_imputemedian`, `ml_imputemode`, `ml_imputeknn` |
| `MLToolKit.ml_knn` | `ml_knn` |
| `MLToolKit.ml_kmeans` | `ml_kmeans` |

An alias works the same way:

```python
import MLToolKit as ml

ml.ml_load
```

`from MLToolKit import load_catalog` counts as an import, but it does not bind the name `MLToolKit`, so completions still need a name that the import actually defines. Comments and strings are ignored:

```python
# import MLToolKit
note = "import MLToolKit"
```

Removing the import, or undoing it, stops the suggestions on the next keystroke. Another open file is not affected. Other Python completions are left alone unless the cursor is on `name.ml_`.

## Develop and update

Regenerate the catalog after adding a snippet in `scripts/catalog_*.py`. The package and the extension both read that generated file.

```powershell
python scripts\build_snippets.py
python scripts\validate_snippets.py
cd vscode-extension
npm install
npm test
cd ..
python -m pytest
```

Run **Run MLToolKit Extension** from the debugger to open an Extension Development Host.

When you change both pieces, bump the version in `pyproject.toml` and `vscode-extension/package.json`, rebuild the catalog, reinstall the package with `python -m pip install -e .`, and reinstall the VSIX or run the install script again. Reload the editor. Pip will not refresh the extension for you.

## Workflows

```powershell
python workflows\classification_workflow.py
python workflows\regression_workflow.py
python workflows\clustering_workflow.py
python workflows\complete_preprocessing_workflow.py
```

Supervised helpers fit imputers, scalers, and encoders on the training split only.

## Layout

```text
MLToolKit/                 import package, catalog.json, implementations
vscode-extension/          TypeScript completion provider
scripts/                   shared catalog source and generator
workflows/                 end-to-end scripts
examples/                  short practical scripts
tests/                     pytest
```

Search [METHOD_INDEX.md](METHOD_INDEX.md) and [snippets/snippet_index.md](snippets/snippet_index.md) for triggers such as `ml_standardscale`.

## Publish to GitHub

```powershell
git init
git add .
git commit -m "Publish MLToolKit package and import-aware completions."
git branch -M main
git remote add origin https://github.com/YOUR_USER/MLToolKit.git
git push -u origin main
```

Or:

```powershell
gh repo create MLToolKit --private --source . --remote origin --push
```

## License

MIT. See [LICENSE](LICENSE).
