# Contributing

Keep preprocessing, EDA, regression, classification, clustering, and evaluation working before adding a model.

## Add a snippet

1. Add a `snippet(...)` entry under `scripts/`.
2. Use a unique `ml-` source trigger. The generator publishes it as `ml_...` for `MLToolKit.ml_...` completions.
3. Put the real imports and executable code in the body. Do not insert `MLToolKit.some_function()`.
4. Point `implementation` at the module that contains the reusable version.
5. Add a category such as `scale` when `MLToolKit.ml_scale` should find that snippet.
6. Run:

```powershell
python scripts\build_snippets.py
python scripts\validate_snippets.py
cd vscode-extension
npm test
cd ..
python -m pytest
```

Placeholder defaults must be valid Python and must not contain `{` or `}`.

## Implementation style

- Fit anything that learns a mean, scale, category list, bin edge, or feature mask on training data only.
- Keep optional libraries inside the function that needs them and raise `ImportError` with the install command.
- Prefer a short function with a docstring over a new framework.
