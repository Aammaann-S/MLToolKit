"""Encoding, transformation, scaling, outliers, selection, splits, and pipelines."""

from catalog_common import snippet

P = "implementations/01_Data_Preprocessing"
F = "implementations/07_Feature_Engineering"

SNIPPETS = [
    snippet("ml-labelencode", "Label encode", "Label-encode one column. Unseen test labels become -1.", """
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        ${1:X_train}[${2:"column"}] = encoder.fit_transform(${1:X_train}[${2:"column"}].astype(str))
        known = set(encoder.classes_)
        ${3:X_test}[${2:"column"}] = ${3:X_test}[${2:"column"}].astype(str).map(
            lambda value: encoder.transform([value])[0] if value in known else -1
        )
        """, ["encode"], f"{P}/categorical_encoding.py"),
    snippet("ml-ordinalencode", "Ordinal encode", "Ordinal-encode columns and keep unknown test values as -1.", """
        from sklearn.preprocessing import OrdinalEncoder

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        ${1:X_train}[${2:columns}] = encoder.fit_transform(${1:X_train}[${2:columns}])
        ${3:X_test}[${2:columns}] = encoder.transform(${3:X_test}[${2:columns}])
        """, ["encode"], f"{P}/categorical_encoding.py"),
    snippet("ml-onehot", "One-hot encode", "One-hot encode categorical columns and ignore unknown test values.", """
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_train = encoder.fit_transform(${1:X_train}[${2:categorical_columns}])
        encoded_test = encoder.transform(${3:X_test}[${2:categorical_columns}])
        """, ["encode"], f"{P}/categorical_encoding.py"),
    snippet("ml-onehotignore", "One-hot with unknown ignored", "Fit one-hot encoding on training categories only.", """
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ${1:X_train_encoded} = encoder.fit_transform(${2:X_train})
        ${3:X_test_encoded} = encoder.transform(${4:X_test})
        """, ["encode"], f"{P}/categorical_encoding.py"),
    snippet("ml-getdummies", "Get dummies", "Create dummies from training columns and align the test frame.", """
        import pandas as pd

        ${1:X_train_encoded} = pd.get_dummies(${2:X_train}, columns=[${3:"column"}])
        ${4:X_test_encoded} = pd.get_dummies(${5:X_test}, columns=[${3:"column"}])
        ${4:X_test_encoded} = ${4:X_test_encoded}.reindex(columns=${1:X_train_encoded}.columns, fill_value=0)
        """, ["encode"], f"{P}/categorical_encoding.py"),
    snippet("ml-frequencyencode", "Frequency encode", "Replace categories with training frequencies.", """
        frequencies = ${1:X_train}[${2:"column"}].value_counts(normalize=True)
        ${1:X_train}[${2:"column"}] = ${1:X_train}[${2:"column"}].map(frequencies)
        ${3:X_test}[${2:"column"}] = ${3:X_test}[${2:"column"}].map(frequencies).fillna(0)
        """, ["encode"], f"{P}/categorical_encoding.py"),
    snippet("ml-logtransform", "Log transform", "Apply log1p to a non-negative column.", """
        import numpy as np

        ${1:df}[${2:"column"}] = np.log1p(${1:df}[${2:"column"}])
        """, ["transform"], f"{P}/numerical_transformation.py"),
    snippet("ml-sqrttransform", "Square root", "Apply a square root to a non-negative column.", """
        import numpy as np

        ${1:df}[${2:"column"}] = np.sqrt(${1:df}[${2:"column"}])
        """, ["transform"], f"{P}/numerical_transformation.py"),
    snippet("ml-powertransform", "Power transformer", "Fit a Yeo-Johnson transformer on the training columns.", """
        from sklearn.preprocessing import PowerTransformer

        transformer = PowerTransformer(method="yeo-johnson")
        ${1:X_train_transformed} = transformer.fit_transform(${2:X_train})
        ${3:X_test_transformed} = transformer.transform(${4:X_test})
        """, ["transform"], f"{P}/numerical_transformation.py"),
    snippet("ml-quantiletransform", "Quantile transformer", "Fit a quantile transformer on the training columns.", """
        from sklearn.preprocessing import QuantileTransformer

        transformer = QuantileTransformer(output_distribution=${1:"normal"}, random_state=${2:42})
        ${3:X_train_transformed} = transformer.fit_transform(${4:X_train})
        ${5:X_test_transformed} = transformer.transform(${6:X_test})
        """, ["transform"], f"{P}/numerical_transformation.py"),
    snippet("ml-binning", "Bin a column", "Bin a numeric column with explicit edges.", """
        import pandas as pd

        ${1:df}[${2:"bin"}] = pd.cut(${1:df}[${3:"column"}], bins=[${4:0}, ${5:18}, ${6:35}, ${7:60}, ${8:100}])
        """, ["transform"], f"{P}/numerical_transformation.py"),
    snippet("ml-kbins", "K-bins discretization", "Learn bin edges on the training data.", """
        from sklearn.preprocessing import KBinsDiscretizer

        discretizer = KBinsDiscretizer(n_bins=${1:4}, encode="ordinal", strategy=${2:"quantile"})
        ${3:X_train_binned} = discretizer.fit_transform(${4:X_train})
        ${5:X_test_binned} = discretizer.transform(${6:X_test})
        """, ["transform"], f"{P}/numerical_transformation.py"),
    snippet("ml-boxcox", "Box-Cox transform", "Fit a Box-Cox transformer on positive training data.", """
        from sklearn.preprocessing import PowerTransformer

        transformer = PowerTransformer(method="box-cox")
        ${1:X_train_transformed} = transformer.fit_transform(${2:X_train})
        ${3:X_test_transformed} = transformer.transform(${4:X_test})
        """, ["transform"], f"{P}/numerical_transformation.py"),
    snippet("ml-polynomialfeatures", "Polynomial features", "Expand polynomial features inside a pipeline fit on training data.", """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import LinearRegression

        pipeline = Pipeline(steps=[
            ("polynomial", PolynomialFeatures(degree=${1:2}, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
        pipeline.fit(${2:X_train}, ${3:y_train})
        predictions = pipeline.predict(${4:X_test})
        """, ["transform"], f"{F}/polynomial_features.py"),
    snippet("ml-standardscale", "StandardScaler", "Standardize features using the training mean and standard deviation.", """
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        ${1:X_train_scaled} = scaler.fit_transform(${2:X_train})
        ${3:X_test_scaled} = scaler.transform(${4:X_test})
        """, ["scale"], f"{P}/feature_scaling.py"),
    snippet("ml-minmaxscale", "MinMaxScaler", "Scale features to a range using training limits.", """
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(${1:0}, ${2:1}))
        ${3:X_train_scaled} = scaler.fit_transform(${4:X_train})
        ${5:X_test_scaled} = scaler.transform(${6:X_test})
        """, ["scale"], f"{P}/feature_scaling.py"),
    snippet("ml-robustscale", "RobustScaler", "Scale with the training median and interquartile range.", """
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
        ${1:X_train_scaled} = scaler.fit_transform(${2:X_train})
        ${3:X_test_scaled} = scaler.transform(${4:X_test})
        """, ["scale"], f"{P}/feature_scaling.py"),
    snippet("ml-maxabsscale", "MaxAbsScaler", "Scale by the training maximum absolute value.", """
        from sklearn.preprocessing import MaxAbsScaler

        scaler = MaxAbsScaler()
        ${1:X_train_scaled} = scaler.fit_transform(${2:X_train})
        ${3:X_test_scaled} = scaler.transform(${4:X_test})
        """, ["scale"], f"{P}/feature_scaling.py"),
    snippet("ml-normalize", "Normalizer", "Scale each row to unit norm. This does not use column statistics.", """
        from sklearn.preprocessing import Normalizer

        normalizer = Normalizer(norm=${1:"l2"})
        ${2:X_normalized} = normalizer.fit_transform(${3:X})
        """, ["scale"], f"{P}/feature_scaling.py"),
    snippet("ml-scalepipeline", "Scale in a pipeline", "Keep scaling inside the estimator pipeline.", """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=${1:1000})),
        ])
        pipeline.fit(${2:X_train}, ${3:y_train})
        predictions = pipeline.predict(${4:X_test})
        """, ["scale", "pipeline"], f"{P}/preprocessing_pipeline.py"),
    snippet("ml-scaletraintest", "Scale train and test", "Fit a scaler on training data and transform both splits.", """
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(${1:X_train})
        ${2:X_train_scaled} = scaler.transform(${1:X_train})
        ${3:X_test_scaled} = scaler.transform(${4:X_test})
        """, ["scale"], f"{P}/feature_scaling.py"),
    snippet("ml-columnscale", "Scale selected columns", "Standardize chosen numeric columns from the training frame.", """
        from sklearn.preprocessing import StandardScaler

        columns = [${1:"feature_0"}, ${2:"feature_1"}]
        scaler = StandardScaler()
        ${3:X_train}[columns] = scaler.fit_transform(${3:X_train}[columns])
        ${4:X_test}[columns] = scaler.transform(${4:X_test}[columns])
        """, ["scale"], f"{P}/feature_scaling.py"),
    snippet("ml-iqr", "IQR bounds", "Compute outlier fences from the training split.", """
        q1 = ${1:X_train}.quantile(0.25)
        q3 = ${1:X_train}.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - ${2:1.5} * iqr
        upper = q3 + ${2:1.5} * iqr
        """, ["outlier"], f"{P}/outlier_handling.py"),
    snippet("ml-zscore", "Z-score", "Compute z-scores from the training mean and standard deviation.", """
        mean = ${1:X_train}.mean()
        std = ${1:X_train}.std(ddof=0).replace(0, 1)
        train_z = (${1:X_train} - mean) / std
        test_z = (${2:X_test} - mean) / std
        """, ["outlier"], f"{P}/outlier_handling.py"),
    snippet("ml-detectoutliers", "Detect IQR outliers", "Mark training rows outside the training fences.", """
        q1 = ${1:X_train}.quantile(0.25)
        q3 = ${1:X_train}.quantile(0.75)
        iqr = q3 - q1
        outlier_mask = ((${1:X_train} < q1 - 1.5 * iqr) | (${1:X_train} > q3 + 1.5 * iqr)).any(axis=1)
        """, ["outlier"], f"{P}/outlier_handling.py"),
    snippet("ml-removeoutliers", "Remove IQR outliers", "Drop training rows outside the training fences.", """
        q1 = ${1:X_train}.quantile(0.25)
        q3 = ${1:X_train}.quantile(0.75)
        iqr = q3 - q1
        mask = ((${1:X_train} < q1 - 1.5 * iqr) | (${1:X_train} > q3 + 1.5 * iqr)).any(axis=1)
        ${1:X_train} = ${1:X_train}.loc[~mask]
        ${2:y_train} = ${2:y_train}.loc[~mask]
        """, ["outlier"], f"{P}/outlier_handling.py"),
    snippet("ml-capoutliers", "Cap outliers", "Clip both splits to fences learned from training data.", """
        lower = ${1:X_train}.quantile(${2:0.05})
        upper = ${1:X_train}.quantile(${3:0.95})
        ${1:X_train} = ${1:X_train}.clip(lower, upper, axis=1)
        ${4:X_test} = ${4:X_test}.clip(lower, upper, axis=1)
        """, ["outlier"], f"{P}/outlier_handling.py"),
    snippet("ml-winsorize", "Winsorize a series", "Winsorize one series with SciPy.", """
        from scipy.stats.mstats import winsorize

        ${1:clipped} = winsorize(${2:series}, limits=(${3:0.05}, ${3:0.05}))
        """, ["outlier"], f"{P}/outlier_handling.py"),
    snippet("ml-outlierboxplot", "Outlier boxplot", "Draw a boxplot for one numeric column.", """
        import matplotlib.pyplot as plt

        plt.boxplot(${1:df}[${2:"column"}].dropna())
        plt.title("Boxplot")
        plt.show()
        """, ["outlier", "eda"], "implementations/02_EDA/visualization.py"),
    snippet("ml-variancethreshold", "Variance threshold", "Drop low-variance features using the training matrix.", """
        from sklearn.feature_selection import VarianceThreshold

        selector = VarianceThreshold(threshold=${1:0.0})
        ${2:X_train_selected} = selector.fit_transform(${3:X_train})
        ${4:X_test_selected} = selector.transform(${5:X_test})
        """, ["select"], f"{P}/feature_selection.py"),
    snippet("ml-selectkbest", "SelectKBest", "Keep the K highest scoring training features.", """
        from sklearn.feature_selection import SelectKBest, f_classif

        selector = SelectKBest(score_func=f_classif, k=${1:5})
        ${2:X_train_selected} = selector.fit_transform(${3:X_train}, ${4:y_train})
        ${5:X_test_selected} = selector.transform(${6:X_test})
        """, ["select"], f"{P}/feature_selection.py"),
    snippet("ml-chi2", "Chi-square selection", "Select K non-negative features with chi-square scores.", """
        from sklearn.feature_selection import SelectKBest, chi2

        selector = SelectKBest(score_func=chi2, k=${1:5})
        ${2:X_train_selected} = selector.fit_transform(${3:X_train}, ${4:y_train})
        ${5:X_test_selected} = selector.transform(${6:X_test})
        """, ["select"], f"{P}/feature_selection.py"),
    snippet("ml-anova", "ANOVA F-test", "Select K features with the ANOVA F-test.", """
        from sklearn.feature_selection import SelectKBest, f_classif

        selector = SelectKBest(score_func=f_classif, k=${1:5})
        ${2:X_train_selected} = selector.fit_transform(${3:X_train}, ${4:y_train})
        ${5:X_test_selected} = selector.transform(${6:X_test})
        """, ["select"], f"{P}/feature_selection.py"),
    snippet("ml-mutualinfo", "Mutual information", "Select K features with mutual information.", """
        from sklearn.feature_selection import SelectKBest, mutual_info_classif

        selector = SelectKBest(score_func=mutual_info_classif, k=${1:5})
        ${2:X_train_selected} = selector.fit_transform(${3:X_train}, ${4:y_train})
        ${5:X_test_selected} = selector.transform(${6:X_test})
        """, ["select"], f"{P}/feature_selection.py"),
    snippet("ml-rfe", "Recursive feature elimination", "Recursively eliminate features on the training split.", """
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression

        selector = RFE(LogisticRegression(max_iter=${1:1000}), n_features_to_select=${2:5})
        ${3:X_train_selected} = selector.fit_transform(${4:X_train}, ${5:y_train})
        ${6:X_test_selected} = selector.transform(${7:X_test})
        """, ["select"], f"{P}/feature_selection.py"),
    snippet("ml-featureimportance", "Tree feature importance", "Rank features with a random forest fit on the training split.", """
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=${1:100}, random_state=${2:42})
        model.fit(${3:X_train}, ${4:y_train})
        importances = model.feature_importances_
        """, ["select"], f"{P}/feature_selection.py"),
    snippet("ml-traintestsplit", "Train test split", "Split features and target with a fixed random state.", """
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            ${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42}
        )
        """, ["split"], f"{P}/train_test_split.py"),
    snippet("ml-stratifiedsplit", "Stratified split", "Preserve class proportions in the test split.", """
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            ${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42}, stratify=${2:y}
        )
        """, ["split"], f"{P}/train_test_split.py"),
    snippet("ml-kfold", "K-fold", "Build shuffled K-fold indices.", """
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=${1:5}, shuffle=True, random_state=${2:42})
        for train_index, test_index in splitter.split(${3:X}):
            X_train, X_test = ${3:X}[train_index], ${3:X}[test_index]
        """, ["split"], f"{P}/train_test_split.py"),
    snippet("ml-stratifiedkfold", "Stratified K-fold", "Build stratified K-fold indices.", """
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=${1:5}, shuffle=True, random_state=${2:42})
        for train_index, test_index in splitter.split(${3:X}, ${4:y}):
            X_train, X_test = ${3:X}[train_index], ${3:X}[test_index]
            y_train, y_test = ${4:y}[train_index], ${4:y}[test_index]
        """, ["split"], f"{P}/train_test_split.py"),
    snippet("ml-timeseriessplit", "Time series split", "Expand the training prefix for each later test block.", """
        from sklearn.model_selection import TimeSeriesSplit

        splitter = TimeSeriesSplit(n_splits=${1:5})
        for train_index, test_index in splitter.split(${2:X}):
            X_train, X_test = ${2:X}[train_index], ${2:X}[test_index]
        """, ["split"], f"{P}/train_test_split.py"),
    snippet("ml-traintestval", "Train validation test split", "Hold out test data before splitting a validation set.", """
        from sklearn.model_selection import train_test_split

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            ${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42}
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=${5:0.25}, random_state=${4:42}
        )
        """, ["split"], f"{P}/train_test_split.py"),
    snippet("ml-splitxy", "Split features and target", "Separate the target column from the predictors.", """
        X = ${1:df}.drop(columns=[${2:"target"}])
        y = ${1:df}[${2:"target"}]
        """, ["split"], f"{P}/train_test_split.py"),
    snippet("ml-randomseed", "Set random seed", "Seed Python and NumPy before a practical.", """
        import random
        import numpy as np

        random.seed(${1:42})
        np.random.seed(${1:42})
        """, ["split", "utility"], "implementations/09_Utilities/random_seed.py"),
    snippet("ml-pipeline", "Pipeline", "Chain preprocessing and a model so every fold fits on training rows only.", """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=${1:1000})),
        ])
        pipeline.fit(${2:X_train}, ${3:y_train})
        predictions = pipeline.predict(${4:X_test})
        """, ["pipeline"], f"{P}/preprocessing_pipeline.py"),
    snippet("ml-columntransformer", "ColumnTransformer", "Impute and encode numeric and categorical columns separately.", """
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        numeric_features = [${1:"age"}, ${2:"income"}]
        categorical_features = [${3:"city"}]
        preprocessor = ColumnTransformer(transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_features),
        ])
        """, ["pipeline"], f"{P}/preprocessing_pipeline.py"),
    snippet("ml-fullpipeline", "Full preprocessing and model pipeline", "Fit imputation, encoding, scaling, and the model on training data only.", """
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        preprocessor = ColumnTransformer(transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), ${1:numeric_features}),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), ${2:categorical_features}),
        ])
        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=${3:1000})),
        ])
        pipeline.fit(${4:X_train}, ${5:y_train})
        predictions = pipeline.predict(${6:X_test})
        """, ["pipeline"], f"{P}/preprocessing_pipeline.py"),
    snippet("ml-preprocesspipeline", "Preprocessing pipeline", "Fit a ColumnTransformer on training features and transform the test features.", """
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        preprocessor = ColumnTransformer(transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), ${1:numeric_features}),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), ${2:categorical_features}),
        ])
        ${3:X_train_prepared} = preprocessor.fit_transform(${4:X_train})
        ${5:X_test_prepared} = preprocessor.transform(${6:X_test})
        """, ["pipeline"], f"{P}/preprocessing_pipeline.py"),
    snippet("ml-pca", "PCA", "Scale and project onto principal components fit on the training split.", """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        pca = PCA(n_components=${1:2}, random_state=${2:42})
        ${3:X_train_pca} = pca.fit_transform(scaler.fit_transform(${4:X_train}))
        ${5:X_test_pca} = pca.transform(scaler.transform(${6:X_test}))
        """, ["transform"], f"{F}/pca.py"),
]
