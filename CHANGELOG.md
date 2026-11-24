# Changelog

## 1.1.0

- Renamed the project to MLToolKit and made `import MLToolKit` the installable package.
- Replaced always-on snippet insertion with a TypeScript extension that activates only after a real MLToolKit import.
- Completions such as `MLToolKit.ml_loadcsv` insert standalone Python, using one catalog shared with the package.

## 1.0.0

- Added the offline VS Code completion provider and more than 100 `ml-` snippets.
- Added reusable preprocessing, EDA, regression, classification, clustering, and evaluation modules.
- Added workflows that train on synthetic data and fit preprocessing on the training split.
- Added optional deep-learning, NLP, time-series, boosting, and deployment modules that do not import their extra packages at startup.
