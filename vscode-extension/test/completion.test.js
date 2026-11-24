"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const detector = require("../out/importDetector");
const engine = require("../out/snippetEngine");

const imported = detector.scanImports("import MLToolKit\n");
assert.strictEqual(imported.active, true);
assert.deepStrictEqual(imported.qualifiers, ["MLToolKit"]);

const aliased = detector.scanImports("import MLToolKit as ml\n");
assert.strictEqual(aliased.active, true);
assert.deepStrictEqual(aliased.qualifiers, ["ml"]);

const fromImport = detector.scanImports("from MLToolKit import load_catalog\n");
assert.strictEqual(fromImport.active, true);
assert.deepStrictEqual(fromImport.qualifiers, []);

const removed = detector.scanImports("import pandas as pd\n");
assert.strictEqual(removed.active, false);
assert.deepStrictEqual(removed.qualifiers, []);

assert.strictEqual(detector.scanImports("# import MLToolKit\n").active, false);
assert.strictEqual(detector.scanImports("note = 'import MLToolKit'\n").active, false);
assert.strictEqual(detector.scanImports('note = "import MLToolKit as ml"\n').active, false);
assert.strictEqual(detector.scanImports('text = """\nimport MLToolKit\n"""\n').active, false);
assert.strictEqual(detector.scanImports("text = r'import MLToolKit'\n").active, false);

const afterString = detector.scanImports("note = 'import MLToolKit'\nimport MLToolKit as toolkit\n");
assert.deepStrictEqual(afterString.qualifiers, ["toolkit"]);

const fileA = detector.scanImports("import MLToolKit as ml\n");
const fileB = detector.scanImports("import os\n");
assert.deepStrictEqual(fileA.qualifiers, ["ml"]);
assert.strictEqual(fileB.active, false);

const multiline = detector.scanImports("from MLToolKit import (\n    load_catalog,\n)\n");
assert.strictEqual(multiline.active, true);

const mixed = detector.scanImports("import os, MLToolKit as ml\n");
assert.deepStrictEqual(mixed.qualifiers, ["ml"]);

const submodule = detector.scanImports("import MLToolKit.preprocessing\n");
assert.deepStrictEqual(submodule.qualifiers, ["MLToolKit"]);

assert.strictEqual(detector.scanImports("import MLToolKitExtra\n").active, false);

const token = detector.findTypedCompletion("MLToolKit.ml_load", "MLToolKit.ml_load".length, ["MLToolKit"]);
assert.deepStrictEqual(token, { qualifier: "MLToolKit", typed: "ml_load", startIndex: 0 });

const aliasToken = detector.findTypedCompletion("value = ml.ml_knn", "value = ml.ml_knn".length, ["ml"]);
assert.strictEqual(aliasToken.qualifier, "ml");
assert.strictEqual(aliasToken.typed, "ml_knn");
assert.strictEqual(aliasToken.startIndex, 8);

assert.strictEqual(detector.findTypedCompletion("MLToolKit.ml_load", 16, []), null);
assert.strictEqual(detector.findTypedCompletion("pandas.read_csv", 15, ["MLToolKit"]), null);
assert.strictEqual(detector.findTypedCompletion("print(MLToolKit)", 16, ["MLToolKit"]), null);
assert.strictEqual(detector.findTypedCompletion("# MLToolKit.ml_load", 20, ["MLToolKit"]).typed, "ml_load");

const catalogPath = path.join(__dirname, "..", "catalog.json");
assert.ok(fs.existsSync(catalogPath), "catalog.json is missing; run scripts/build_snippets.py");
const catalog = engine.parseCatalog(fs.readFileSync(catalogPath, "utf8"));
const triggers = (typed) => engine.filterSnippets(catalog.snippets, typed).map((item) => item.trigger);

for (const trigger of ["ml_loaddataset", "ml_loadcsv", "ml_loadexcel", "ml_loadjson"]) {
  assert.ok(triggers("ml_load").includes(trigger), trigger);
}
for (const trigger of ["ml_standardscale", "ml_minmaxscale", "ml_robustscale", "ml_maxabsscale", "ml_normalize", "ml_scalepipeline"]) {
  assert.ok(triggers("ml_scale").includes(trigger), trigger);
}
assert.ok(triggers("ml_impute").includes("ml_imputemean"));
assert.ok(triggers("ml_knn").includes("ml_knn"));
assert.ok(triggers("ml_kmeans").includes("ml_kmeans"));
assert.ok(triggers("ml_linear").includes("ml_linearreg"));
assert.ok(!triggers("ml_standard").includes("ml_minmaxscale"));

const csv = catalog.snippets.find((item) => item.trigger === "ml_loadcsv");
assert.ok(csv.body.includes("pd.read_csv"));
assert.ok(!csv.body.includes("MLToolKit."));
assert.ok(csv.placeholders.length > 0);
assert.strictEqual(csv.category, "load");

console.log("MLToolKit completion tests passed");
