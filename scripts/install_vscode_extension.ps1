# Compile and install the MLToolKit extension into VS Code or Cursor.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$extension = Join-Path $repo "vscode-extension"
$catalog = Join-Path $repo "MLToolKit\catalog.json"

if (-not (Test-Path $catalog)) {
    python (Join-Path $repo "scripts\build_snippets.py")
}
Push-Location $extension
npm install
npm run compile
Pop-Location

$editors = @(
    (Join-Path $env:USERPROFILE ".vscode\extensions"),
    (Join-Path $env:USERPROFILE ".cursor\extensions")
)
foreach ($extensionsRoot in $editors) {
    if (-not (Test-Path (Split-Path $extensionsRoot))) {
        continue
    }
    $destination = Join-Path $extensionsRoot "local.mltoolkit-1.1.0"
    if (Test-Path $destination) {
        Remove-Item $destination -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    Copy-Item (Join-Path $extension "package.json") $destination
    Copy-Item (Join-Path $extension "catalog.json") $destination
    Copy-Item (Join-Path $extension "out") (Join-Path $destination "out") -Recurse
    Write-Output "Installed to $destination"
}

Write-Output "Reload VS Code or Cursor. pip does not install this extension."
