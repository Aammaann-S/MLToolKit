"""Write the shared MLToolKit snippet catalog used by the package and the extension."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog_data import SNIPPETS as DATA
from catalog_extra import SNIPPETS as EXTRA
from catalog_models import SNIPPETS as MODELS
from catalog_prepare import SNIPPETS as PREPARE

VERSION = "1.1.0"
REQUIRED = [
    "ml-loaddataset",
    "ml-readdataset",
    "ml-loadexcel",
    "ml-loadjson",
    "ml-loadmultiple",
    "ml-loadcsv",
    "ml-loadcsvpath",
    "ml-loadsql",
    "ml-standardscale",
    "ml-minmaxscale",
    "ml-robustscale",
    "ml-maxabsscale",
    "ml-normalize",
    "ml-scalepipeline",
    "ml-imputemean",
    "ml-imputemedian",
    "ml-imputemode",
    "ml-imputeknn",
    "ml-imputeconstant",
    "ml-imputeforwardfill",
    "ml-imputebackwardfill",
    "ml-dropmissing",
    "ml-checkmissing",
    "ml-labelencode",
    "ml-ordinalencode",
    "ml-onehot",
    "ml-getdummies",
    "ml-linearreg",
    "ml-multilinearreg",
    "ml-polynomialreg",
    "ml-ridgereg",
    "ml-lassoreg",
    "ml-logisticreg",
    "ml-knn",
    "ml-naivebayes",
    "ml-decisiontreeclassifier",
    "ml-randomforestclassifier",
    "ml-svmclassifier",
    "ml-kmeans",
    "ml-hierarchical",
    "ml-dbscan",
    "ml-elbow",
    "ml-silhouette",
    "ml-confusionmatrix",
]


def public_name(value: str) -> str:
    return value.replace("-", "_")


def aliases_for(item) -> list[str]:
    found = []
    for category in item.categories:
        alias = f"ml-{category}"
        if alias != item.prefix and alias not in found:
            found.append(alias)
    for alias in item.extra_aliases:
        if alias != item.prefix and alias not in found:
            found.append(alias)
    return [public_name(alias) for alias in found]


def import_lines(body: str) -> list[str]:
    lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            lines.append(stripped)
    return lines


def placeholders(body: str) -> list[dict[str, str]]:
    return [
        {"index": index, "default": default}
        for index, default in re.findall(r"\$\{(\d+):([^{}]*)\}", body)
    ]


def markdown_table(items) -> str:
    lines = ["| Trigger | Description | Implementation |", "| --- | --- | --- |"]
    for item in items:
        lines.append(
            f"| `{public_name(item.prefix)}` | {item.description} | `MLToolKit/{item.implementation}` |"
        )
    return "\n".join(lines)


def main() -> None:
    snippets = [*DATA, *PREPARE, *MODELS, *EXTRA]
    prefixes = [item.prefix for item in snippets]
    titles = [item.title for item in snippets]
    duplicate_prefixes = sorted({prefix for prefix in prefixes if prefixes.count(prefix) > 1})
    duplicate_titles = sorted({title for title in titles if titles.count(title) > 1})
    if duplicate_prefixes or duplicate_titles:
        raise SystemExit(f"Duplicate snippet keys: prefixes={duplicate_prefixes} titles={duplicate_titles}")
    missing = [prefix for prefix in REQUIRED if prefix not in prefixes]
    if missing:
        raise SystemExit(f"Missing required triggers: {missing}")
    if len(snippets) < 100:
        raise SystemExit(f"Expected at least 100 snippets, found {len(snippets)}")

    by_category: dict[str, list[str]] = defaultdict(list)
    for item in snippets:
        by_category[item.categories[0]].append(public_name(item.prefix))

    records = []
    for item in snippets:
        trigger = public_name(item.prefix)
        category = item.categories[0]
        records.append(
            {
                "trigger": trigger,
                "displayName": item.title,
                "description": item.description,
                "category": category,
                "categories": list(item.categories),
                "aliases": aliases_for(item),
                "imports": import_lines(item.body),
                "body": item.body,
                "placeholders": placeholders(item.body),
                "implementation": f"MLToolKit/{item.implementation}",
                "variants": [name for name in by_category[category] if name != trigger],
            }
        )

    catalog = {"package": "MLToolKit", "version": VERSION, "snippets": records}
    payload = json.dumps(catalog, indent=2) + "\n"
    destinations = [
        ROOT / "MLToolKit" / "catalog.json",
        ROOT / "vscode-extension" / "catalog.json",
    ]
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")

    grouped: dict[str, list] = defaultdict(list)
    for item in snippets:
        grouped[item.categories[0]].append(item)
    index_sections = [
        "# Snippet index",
        "",
        f"{len(snippets)} snippets. Type `MLToolKit.ml_` after importing the package.",
        "",
    ]
    method_sections = [
        "# Method index",
        "",
        "Search this file for a trigger such as `ml_standardscale`.",
        "",
    ]
    for category in sorted(grouped):
        block = f"## {category}\n\n" + markdown_table(sorted(grouped[category], key=lambda item: item.prefix))
        index_sections.extend([block, ""])
        method_sections.extend([block, ""])
    (ROOT / "snippets" / "snippet_index.md").write_text("\n".join(index_sections) + "\n", encoding="utf-8")
    (ROOT / "METHOD_INDEX.md").write_text("\n".join(method_sections) + "\n", encoding="utf-8")

    algorithm_categories = ["regression", "classification", "cluster", "advanced", "deep", "nlp", "series"]
    algorithm_lines = [
        "# Algorithm index",
        "",
        "Algorithms inserted by the MLToolKit extension or called from the package modules.",
        "",
    ]
    for category in algorithm_categories:
        algorithm_lines.extend(
            [f"## {category}", "", markdown_table(sorted(grouped[category], key=lambda item: item.prefix)), ""]
        )
    (ROOT / "ALGORITHM_INDEX.md").write_text("\n".join(algorithm_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} snippets to {destinations[0]}")


if __name__ == "__main__":
    main()
