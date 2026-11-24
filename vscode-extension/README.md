# MLToolKit VS Code extension

This extension watches the active Python document. After a real `import MLToolKit` or `import MLToolKit as ml`, typing `MLToolKit.ml_` or `ml.ml_` offers categorized snippets. Accepting one inserts the Python implementation, not a call into the package.

Imports inside comments and strings are ignored. Each open file is scanned on its own, including after undo and redo. The scanner does not execute Python.

## Develop

```powershell
cd vscode-extension
npm install
npm test
```

In VS Code or Cursor, open this repository and run the **Run MLToolKit Extension** launch configuration. That opens an Extension Development Host.

## Package a VSIX

```powershell
cd vscode-extension
npx vsce package --allow-missing-repository
code --install-extension mltoolkit-1.1.0.vsix
```

Installing the Python package with pip does not install this extension. Update the extension by rebuilding the VSIX, or by running `scripts/install_vscode_extension.ps1` again, then reload the editor.
