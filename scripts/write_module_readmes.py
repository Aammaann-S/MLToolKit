"""Write a short README in each implementation folder."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "MLToolKit" / "implementations"

for folder in sorted(path for path in ROOT.iterdir() if path.is_dir()):
    files = [path.relative_to(folder).as_posix() for path in sorted(folder.rglob("*.py"))]
    title = folder.name.replace("_", " ")
    lines = [
        f"# {title}",
        "",
        "Reusable functions for this part of the toolkit. Snippet bodies stay standalone; these modules are the callable versions.",
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- `{name}`" for name in files)
    lines.extend(["", "Matching snippet triggers are listed in `snippets/snippet_index.md` and `METHOD_INDEX.md`.", ""])
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(folder.name, len(files))
