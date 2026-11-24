"""Validate the shared catalog used by the package and the VS Code extension."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATHS = [
    ROOT / "MLToolKit" / "catalog.json",
    ROOT / "vscode-extension" / "catalog.json",
]
REQUIRED_QUERIES = {
    "ml_load": ["ml_loaddataset", "ml_readdataset", "ml_loadexcel", "ml_loadjson", "ml_loadmultiple"],
    "ml_scale": ["ml_standardscale", "ml_minmaxscale", "ml_robustscale", "ml_maxabsscale", "ml_normalize", "ml_scalepipeline"],
    "ml_impute": ["ml_imputemean", "ml_imputemedian", "ml_imputemode", "ml_imputeknn", "ml_imputeconstant", "ml_imputeforwardfill", "ml_imputebackwardfill"],
    "ml_knn": ["ml_knn"],
    "ml_kmeans": ["ml_kmeans"],
    "ml_linearreg": ["ml_linearreg"],
    "ml_logisticreg": ["ml_logisticreg"],
}


def materialize(body: str) -> str:
    body = re.sub(r"\$\{(\d+)\|([^{}|]*)\|\}", lambda match: match.group(2).split(",")[0], body)
    body = re.sub(r"\$\{(\d+):([^{}]*)\}", lambda match: match.group(2), body)
    if re.search(r"\$\{\d+:", body):
        raise ValueError("A placeholder default contains a brace or was not closed")
    body = re.sub(r"\$(\d+)", "None", body)
    return body


def matching_triggers(snippets: list[dict], typed: str) -> list[str]:
    query = typed.lower()
    matches = []
    for snippet in snippets:
        keys = [snippet["trigger"], *snippet.get("aliases", [])]
        if any(str(key).lower().startswith(query) for key in keys):
            matches.append(snippet["trigger"])
    return sorted(matches)


def main() -> None:
    payloads = [path.read_text(encoding="utf-8") for path in CATALOG_PATHS]
    if payloads[0] != payloads[1]:
        raise SystemExit("Package catalog and extension catalog differ")
    document = json.loads(payloads[0])
    snippets = document["snippets"]
    if len(snippets) < 100:
        raise SystemExit(f"Expected at least 100 snippets, found {len(snippets)}")
    triggers = []
    for snippet in snippets:
        trigger = snippet["trigger"]
        if not re.fullmatch(r"ml_[a-z0-9]+", trigger):
            raise SystemExit(f"{trigger} does not match ml_[a-z0-9]+")
        if not snippet.get("description") or not snippet.get("displayName") or not snippet.get("category"):
            raise SystemExit(f"{trigger} is missing catalog metadata")
        implementation = snippet.get("implementation")
        if not implementation or not (ROOT / implementation).is_file():
            raise SystemExit(f"{trigger} points at a missing implementation: {implementation}")
        if "MLToolKit." in snippet["body"]:
            raise SystemExit(f"{trigger} inserts a toolkit call instead of standalone code")
        try:
            ast.parse(materialize(snippet["body"]))
        except SyntaxError as exc:
            raise SystemExit(f"{trigger} is not valid Python after placeholder substitution: {exc}") from exc
        triggers.append(trigger)
    duplicates = sorted({trigger for trigger in triggers if triggers.count(trigger) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate triggers: {duplicates}")
    for query, expected in REQUIRED_QUERIES.items():
        found = matching_triggers(snippets, query)
        missing = [trigger for trigger in expected if trigger not in found]
        if missing:
            raise SystemExit(f"Query {query} did not find {missing}")
    csv = next(item for item in snippets if item["trigger"] == "ml_loadcsv")
    if "pd.read_csv" not in csv["body"]:
        raise SystemExit("ml_loadcsv does not insert pandas read_csv code")
    if not csv["placeholders"]:
        raise SystemExit("ml_loadcsv has no placeholders")
    index = (ROOT / "snippets" / "snippet_index.md").read_text(encoding="utf-8")
    for trigger in triggers:
        if f"`{trigger}`" not in index:
            raise SystemExit(f"{trigger} is missing from the snippet index")
    print(f"Validated {len(snippets)} shared snippets")


if __name__ == "__main__":
    main()
