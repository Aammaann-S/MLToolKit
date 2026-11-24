# Roadmap

The core library is the part that must stay complete. The items below are later additions, not missing pieces of the current practical workflow.

## Planned

- Transformer text classification that downloads no weights during import.
- A Flask prediction app beside the FastAPI template.
- A Docker example for the prediction service.
- A simple data-drift report that compares training and scoring distributions.
- Transfer-learning image stubs that fail clearly when the backbone weights are absent.
- Notebook walkthroughs for a full college practical, generated from the existing snippets.

## Already present and optional

- XGBoost, LightGBM, and CatBoost trainers.
- SHAP and LIME wrappers.
- Small PyTorch and Keras ANN, CNN, and RNN trainers.
- ARIMA fitting through statsmodels.
- MLflow run logging, with a local JSON log that works without MLflow.
