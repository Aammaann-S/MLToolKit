"""EDA, advanced ML, NLP, time series, deployment, and utility snippets."""

from catalog_common import snippet

EDA = "implementations/02_EDA"
ADV = "implementations/08_Advanced_ML"

SNIPPETS = [
    snippet("ml-groupby", "Group by", "Aggregate one column by another.", """
        ${1:df}.groupby(${2:"group"})[${3:"value"}].mean()
        """, ["eda"], f"{EDA}/data_summary.py"),
    snippet("ml-agg", "Aggregate", "Group and aggregate several statistics.", """
        ${1:df}.groupby(${2:"group"}).agg({${3:"value"}: ["mean", "median", "count"]})
        """, ["eda"], f"{EDA}/descriptive_statistics.py"),
    snippet("ml-merge", "Merge", "Merge two frames on a key.", """
        merged = ${1:left}.merge(${2:right}, on=${3:"id"}, how=${4:"inner"})
        """, ["eda"], f"{EDA}/data_summary.py"),
    snippet("ml-join", "Join on index", "Join two frames by index.", """
        joined = ${1:left}.join(${2:right}, how=${3:"left"})
        """, ["eda"], f"{EDA}/data_summary.py"),
    snippet("ml-concat", "Concatenate", "Stack frames by rows.", """
        import pandas as pd

        combined = pd.concat([${1:first}, ${2:second}], ignore_index=True)
        """, ["eda"], f"{EDA}/data_summary.py"),
    snippet("ml-pivot", "Pivot table", "Build a pivot table.", """
        import pandas as pd

        table = pd.pivot_table(${1:df}, index=${2:"group"}, columns=${3:"category"}, values=${4:"value"}, aggfunc=${5:"mean"})
        """, ["eda"], f"{EDA}/data_summary.py"),
    snippet("ml-crosstab", "Crosstab", "Cross-tabulate two categorical columns.", """
        import pandas as pd

        table = pd.crosstab(${1:df}[${2:"row"}], ${1:df}[${3:"column"}])
        """, ["eda"], f"{EDA}/categorical_analysis.py"),
    snippet("ml-correlation", "Correlation", "Pearson correlation of numeric columns.", """
        ${1:df}.select_dtypes(include="number").corr(method=${2:"pearson"})
        """, ["eda"], f"{EDA}/correlation.py"),
    snippet("ml-corrheatmap", "Correlation heatmap", "Draw a correlation heatmap.", """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.heatmap(${1:df}.select_dtypes(include="number").corr(), annot=True, cmap="vlag", center=0)
        plt.title("Correlation")
        plt.show()
        """, ["eda"], f"{EDA}/visualization.py"),
    snippet("ml-histogram", "Histogram", "Draw a histogram.", """
        import matplotlib.pyplot as plt

        plt.hist(${1:df}[${2:"column"}].dropna(), bins=${3:30})
        plt.title("Histogram")
        plt.show()
        """, ["eda"], f"{EDA}/visualization.py"),
    snippet("ml-boxplot", "Boxplot", "Draw a boxplot.", """
        import matplotlib.pyplot as plt

        plt.boxplot(${1:df}[${2:"column"}].dropna())
        plt.title("Boxplot")
        plt.show()
        """, ["eda"], f"{EDA}/visualization.py"),
    snippet("ml-scatter", "Scatter plot", "Draw a scatter plot.", """
        import matplotlib.pyplot as plt

        plt.scatter(${1:df}[${2:"x"}], ${1:df}[${3:"y"}])
        plt.xlabel(${2:"x"})
        plt.ylabel(${3:"y"})
        plt.show()
        """, ["eda"], f"{EDA}/visualization.py"),
    snippet("ml-pairplot", "Pair plot", "Draw a Seaborn pair plot.", """
        import seaborn as sns

        sns.pairplot(${1:df}, hue=${2:"target"})
        """, ["eda"], f"{EDA}/visualization.py"),
    snippet("ml-countplot", "Count plot", "Draw a Seaborn count plot.", """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.countplot(data=${1:df}, x=${2:"column"})
        plt.show()
        """, ["eda"], f"{EDA}/visualization.py"),
    snippet("ml-distplot", "Distribution plot", "Draw a histogram with a density curve.", """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.histplot(data=${1:df}, x=${2:"column"}, kde=True)
        plt.show()
        """, ["eda"], f"{EDA}/visualization.py"),
    snippet("ml-missingheatmap", "Missing-value heatmap", "Show which cells are missing.", """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.heatmap(${1:df}.isna(), cbar=False)
        plt.title("Missing values")
        plt.show()
        """, ["eda", "missing"], f"{EDA}/visualization.py"),
    snippet("ml-syntheticclass", "Synthetic classification data", "Build a classification table with scikit-learn.", """
        import pandas as pd
        from sklearn.datasets import make_classification

        features, target = make_classification(n_samples=${1:200}, n_features=${2:4}, n_classes=${3:2}, random_state=${4:42})
        df = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(features.shape[1])])
        df["target"] = target
        """, ["utility"], "implementations/09_Utilities/synthetic_datasets.py"),
    snippet("ml-syntheticreg", "Synthetic regression data", "Build a regression table with scikit-learn.", """
        import pandas as pd
        from sklearn.datasets import make_regression

        features, target = make_regression(n_samples=${1:200}, n_features=${2:3}, noise=${3:10.0}, random_state=${4:42})
        df = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(features.shape[1])])
        df["target"] = target
        """, ["utility"], "implementations/09_Utilities/synthetic_datasets.py"),
    snippet("ml-syntheticcluster", "Synthetic clusters", "Build a blob cluster table.", """
        import pandas as pd
        from sklearn.datasets import make_blobs

        features, labels = make_blobs(n_samples=${1:180}, centers=${2:3}, random_state=${3:42})
        df = pd.DataFrame(features, columns=["feature_0", "feature_1"])
        df["cluster"] = labels
        """, ["utility", "cluster"], "implementations/09_Utilities/synthetic_datasets.py"),
    snippet("ml-setseed", "Set seed", "Set the random, NumPy, and PyTorch seeds when PyTorch is installed.", """
        import random
        import numpy as np

        SEED = ${1:42}
        random.seed(SEED)
        np.random.seed(SEED)
        """, ["utility"], "implementations/09_Utilities/random_seed.py"),
    snippet("ml-smote", "SMOTE", "Resample the training split with SMOTE. Requires imbalanced-learn.", """
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=${1:42})
        X_train_resampled, y_train_resampled = sampler.fit_resample(${2:X_train}, ${3:y_train})
        """, ["advanced"], f"{ADV}/imbalanced.py"),
    snippet("ml-classweight", "Balanced class weight", "Fit logistic regression with balanced class weights.", """
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=${1:1000}, class_weight="balanced", random_state=${2:42})
        model.fit(${3:X_train}, ${4:y_train})
        predictions = model.predict(${5:X_test})
        """, ["advanced", "classification"], f"{ADV}/imbalanced.py"),
    snippet("ml-xgboost", "XGBoost classifier", "Fit XGBoost on a stratified split. Requires xgboost.", """
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            ${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42}, stratify=${2:y}
        )
        model = XGBClassifier(n_estimators=${5:100}, max_depth=${6:3}, random_state=${4:42})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        """, ["advanced"], f"{ADV}/boosting.py"),
    snippet("ml-lightgbm", "LightGBM classifier", "Fit LightGBM on the training split. Requires lightgbm.", """
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(n_estimators=${1:100}, random_state=${2:42}, verbose=-1)
        model.fit(${3:X_train}, ${4:y_train})
        predictions = model.predict(${5:X_test})
        """, ["advanced"], f"{ADV}/boosting.py"),
    snippet("ml-catboost", "CatBoost classifier", "Fit CatBoost on the training split. Requires catboost.", """
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(iterations=${1:100}, depth=${2:4}, verbose=0, random_state=${3:42})
        model.fit(${4:X_train}, ${5:y_train})
        predictions = model.predict(${6:X_test})
        """, ["advanced"], f"{ADV}/boosting.py"),
    snippet("ml-adaboost", "AdaBoost", "Fit an AdaBoost classifier on the training split.", """
        from sklearn.ensemble import AdaBoostClassifier

        model = AdaBoostClassifier(n_estimators=${1:50}, random_state=${2:42})
        model.fit(${3:X_train}, ${4:y_train})
        predictions = model.predict(${5:X_test})
        """, ["advanced"], f"{ADV}/ensemble_learning.py"),
    snippet("ml-extratrees", "Extra trees", "Fit an extra-trees classifier on the training split.", """
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(n_estimators=${1:100}, random_state=${2:42})
        model.fit(${3:X_train}, ${4:y_train})
        predictions = model.predict(${5:X_test})
        """, ["advanced"], f"{ADV}/ensemble_learning.py"),
    snippet("ml-isolationforest", "Isolation Forest", "Flag anomalies with Isolation Forest.", """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(${1:X})
        model = IsolationForest(contamination=${2:0.05}, random_state=${3:42})
        labels = model.fit_predict(X_scaled)
        """, ["advanced"], f"{ADV}/anomaly_detection.py"),
    snippet("ml-lof", "Local outlier factor", "Flag anomalies with local outlier factor.", """
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(${1:X})
        labels = LocalOutlierFactor(n_neighbors=${2:20}, contamination=${3:0.05}).fit_predict(X_scaled)
        """, ["advanced"], f"{ADV}/anomaly_detection.py"),
    snippet("ml-gmm", "Gaussian mixture", "Fit a Gaussian mixture on scaled features.", """
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(${1:X})
        model = GaussianMixture(n_components=${2:3}, random_state=${3:42})
        labels = model.fit_predict(X_scaled)
        """, ["advanced", "cluster"], f"{ADV}/advanced_clustering.py"),
    snippet("ml-spectral", "Spectral clustering", "Fit spectral clustering on scaled features.", """
        from sklearn.cluster import SpectralClustering
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(${1:X})
        labels = SpectralClustering(n_clusters=${2:3}, random_state=${3:42}, affinity="nearest_neighbors").fit_predict(X_scaled)
        """, ["advanced", "cluster"], f"{ADV}/advanced_clustering.py"),
    snippet("ml-voting", "Voting classifier", "Combine logistic regression, a tree, and KNN.", """
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier

        model = VotingClassifier(estimators=[
            ("logistic", LogisticRegression(max_iter=1000)),
            ("tree", DecisionTreeClassifier(max_depth=4, random_state=${1:42})),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ])
        model.fit(${2:X_train}, ${3:y_train})
        predictions = model.predict(${4:X_test})
        """, ["advanced"], f"{ADV}/ensemble_learning.py"),
    snippet("ml-textclean", "Clean text", "Lowercase text and remove URLs and punctuation.", """
        import re

        text = str(${1:text}).lower()
        text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
        text = re.sub(r"[^a-z0-9\\s]", " ", text)
        text = re.sub(r"\\s+", " ", text).strip()
        """, ["nlp"], "implementations/11_NLP/text_preprocessing.py"),
    snippet("ml-tfidf", "TF-IDF", "Fit TF-IDF on training documents and transform the test documents.", """
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=${1:5000}, ngram_range=(1, 2))
        X_train_text = vectorizer.fit_transform(${2:train_text})
        X_test_text = vectorizer.transform(${3:test_text})
        """, ["nlp"], "implementations/11_NLP/tfidf.py"),
    snippet("ml-bow", "Bag of words", "Fit word counts on training documents.", """
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(max_features=${1:5000})
        X_train_text = vectorizer.fit_transform(${2:train_text})
        X_test_text = vectorizer.transform(${3:test_text})
        """, ["nlp"], "implementations/11_NLP/bag_of_words.py"),
    snippet("ml-textclassify", "Classify text", "Classify text with TF-IDF and logistic regression in one pipeline.", """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline

        X_train, X_test, y_train, y_test = train_test_split(
            ${1:texts}, ${2:labels}, test_size=${3:0.2}, random_state=${4:42}, stratify=${2:labels}
        )
        model = Pipeline(steps=[
            ("tfidf", TfidfVectorizer(max_features=${5:5000})),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        """, ["nlp"], "implementations/11_NLP/text_classification.py"),
    snippet("ml-resample", "Resample time series", "Resample a datetime-indexed series.", """
        ${1:series} = ${1:series}.resample(${2:"D"}).mean()
        """, ["series"], "implementations/12_Time_Series/temporal_preprocessing.py"),
    snippet("ml-lagfeatures", "Lag features", "Add lagged copies of a series.", """
        column = ${2:"value"}
        ${1:df}[f"{column}_lag_1"] = ${1:df}[column].shift(1)
        ${1:df}[f"{column}_lag_2"] = ${1:df}[column].shift(2)
        """, ["series"], "implementations/12_Time_Series/feature_extraction.py"),
    snippet("ml-rolling", "Rolling statistics", "Rolling mean of past values, excluding the current row.", """
        column = ${2:"value"}
        past = ${1:df}[column].shift(1)
        ${1:df}[f"{column}_roll_mean"] = past.rolling(${3:3}).mean()
        """, ["series"], "implementations/12_Time_Series/temporal_preprocessing.py"),
    snippet("ml-arima", "ARIMA forecast", "Fit an ARIMA model and forecast the next steps.", """
        from statsmodels.tsa.arima.model import ARIMA

        fitted = ARIMA(${1:series}, order=(${2:1}, ${3:1}, ${4:1})).fit()
        forecast = fitted.forecast(steps=${5:3})
        """, ["series"], "implementations/12_Time_Series/forecasting.py"),
    snippet("ml-fastapi", "FastAPI prediction route", "Serve a joblib model with FastAPI. Requires fastapi and pydantic.", """
        import joblib
        from fastapi import FastAPI
        from pydantic import BaseModel

        app = FastAPI()
        model = joblib.load(${1:"model.joblib"})

        class Features(BaseModel):
            values: list[float]

        @app.post("/predict")
        def predict(payload: Features):
            prediction = model.predict([payload.values])
            return {"prediction": prediction.tolist()}
        """, ["deploy"], "implementations/13_Deployment_MLOps/fastapi_template.py"),
    snippet("ml-mlflow", "Log metrics locally", "Write parameters and metrics to a JSON file without a tracking server.", """
        import json
        from pathlib import Path

        payload = {"params": ${1:params}, "metrics": ${2:metrics}}
        Path(${3:"run.json"}).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        """, ["deploy"], "implementations/13_Deployment_MLOps/mlflow_tracking.py"),
    snippet("ml-torchann", "PyTorch ANN", "Train a small dropout network. Requires torch.", """
        import torch
        from torch import nn

        features = torch.tensor(${1:X}, dtype=torch.float32)
        target = torch.tensor(${2:y}, dtype=torch.long)
        model = nn.Sequential(
            nn.Linear(features.shape[1], ${3:16}),
            nn.ReLU(),
            nn.Dropout(${4:0.2}),
            nn.Linear(${3:16}, int(target.max()) + 1),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=${5:0.01})
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(${6:20}):
            optimizer.zero_grad()
            loss = loss_fn(model(features), target)
            loss.backward()
            optimizer.step()
        """, ["deep"], "implementations/10_Deep_Learning/pytorch/ann.py"),
    snippet("ml-kerasann", "Keras ANN", "Train a small Keras network. Requires tensorflow.", """
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(${1:X}.shape[1],)),
            tf.keras.layers.Dense(${2:16}, activation="relu"),
            tf.keras.layers.Dropout(${3:0.2}),
            tf.keras.layers.Dense(len(set(${4:y})), activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(${1:X}, ${4:y}, epochs=${5:20}, verbose=0)
        """, ["deep"], "implementations/10_Deep_Learning/tensorflow/ann.py"),
    snippet("ml-ratiofeature", "Ratio feature", "Divide one numeric column by another.", """
        ${1:df}[${2:"ratio"}] = ${1:df}[${3:"numerator"}] / ${1:df}[${4:"denominator"}].replace(0, float("nan"))
        """, ["transform"], "implementations/07_Feature_Engineering/feature_creation.py"),
    snippet("ml-datetimeparts", "Datetime parts", "Add year, month, day, and weekday columns.", """
        import pandas as pd

        values = pd.to_datetime(${1:df}[${2:"date"}], errors="coerce")
        ${1:df}["year"] = values.dt.year
        ${1:df}["month"] = values.dt.month
        ${1:df}["day"] = values.dt.day
        ${1:df}["day_of_week"] = values.dt.day_of_week
        """, ["transform"], "implementations/07_Feature_Engineering/feature_creation.py"),
]
