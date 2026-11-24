"""Regression, classification, clustering, evaluation, tuning, and persistence snippets."""

from catalog_common import snippet

R = "implementations/03_Regression"
C = "implementations/04_Classification"
K = "implementations/05_Clustering"
E = "implementations/06_Model_Evaluation"

SNIPPETS = [
    snippet("ml-linearreg", "Simple linear regression", "Fit one-feature linear regression after splitting.", """
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(${1:"data.csv"})
        X = df[[${2:"feature"}]]
        y = df[${3:"target"}]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${4:0.2}, random_state=${5:42})
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("MAE", mean_absolute_error(y_test, predictions))
        print("MSE", mean_squared_error(y_test, predictions))
        print("RMSE", mean_squared_error(y_test, predictions) ** 0.5)
        print("R2", r2_score(y_test, predictions))
        """, ["regression"], f"{R}/simple_linear_regression.py"),
    snippet("ml-multilinearreg", "Multiple linear regression", "Fit linear regression with several predictors.", """
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(${1:"data.csv"})
        X = df[${2:["feature_0", "feature_1"]}]
        y = df[${3:"target"}]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${4:0.2}, random_state=${5:42})
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("MAE", mean_absolute_error(y_test, predictions))
        print("RMSE", mean_squared_error(y_test, predictions) ** 0.5)
        print("R2", r2_score(y_test, predictions))
        """, ["regression"], f"{R}/multiple_linear_regression.py"),
    snippet("ml-polynomialreg", "Polynomial regression", "Expand polynomial features inside a pipeline fit on the training split.", """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        X_train, X_test, y_train, y_test = train_test_split(${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42})
        model = Pipeline(steps=[
            ("polynomial", PolynomialFeatures(degree=${5:2}, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("R2", r2_score(y_test, predictions))
        """, ["regression"], f"{R}/polynomial_regression.py"),
    snippet("ml-ridgereg", "Ridge regression", "Fit ridge regression on the training split.", """
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42})
        model = Ridge(alpha=${5:1.0}, random_state=${4:42})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("RMSE", mean_squared_error(y_test, predictions) ** 0.5)
        print("R2", r2_score(y_test, predictions))
        """, ["regression"], f"{R}/ridge_regression.py"),
    snippet("ml-lassoreg", "Lasso regression", "Fit lasso regression on the training split.", """
        from sklearn.linear_model import Lasso
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42})
        model = Lasso(alpha=${5:0.1}, random_state=${4:42}, max_iter=${6:10000})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("RMSE", mean_squared_error(y_test, predictions) ** 0.5)
        print("R2", r2_score(y_test, predictions))
        """, ["regression"], f"{R}/lasso_regression.py"),
    snippet("ml-elasticnet", "Elastic Net", "Fit Elastic Net on the training split.", """
        from sklearn.linear_model import ElasticNet
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42})
        model = ElasticNet(alpha=${5:0.1}, l1_ratio=${6:0.5}, random_state=${4:42}, max_iter=${7:10000})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("R2", r2_score(y_test, predictions))
        """, ["regression"], f"{R}/elasticnet_regression.py"),
    snippet("ml-decisiontreereg", "Decision tree regressor", "Fit a decision tree regressor on the training split.", """
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor

        X_train, X_test, y_train, y_test = train_test_split(${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42})
        model = DecisionTreeRegressor(max_depth=${5:4}, random_state=${4:42})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("R2", r2_score(y_test, predictions))
        """, ["regression"], f"{R}/decision_tree_regressor.py"),
    snippet("ml-randomforestreg", "Random forest regressor", "Fit a random forest regressor on the training split.", """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42})
        model = RandomForestRegressor(n_estimators=${5:100}, random_state=${4:42})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("R2", r2_score(y_test, predictions))
        """, ["regression"], f"{R}/random_forest_regressor.py"),
    snippet("ml-svr", "Support vector regression", "Fit SVR on features scaled inside a pipeline.", """
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        X_train, X_test, y_train, y_test = train_test_split(${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42})
        model = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("regressor", SVR(kernel=${5:"rbf"}, C=${6:1.0})),
        ])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("R2", r2_score(y_test, predictions))
        """, ["regression"], f"{R}/regression_evaluation.py"),
    snippet("ml-gradientboostingreg", "Gradient boosting regressor", "Fit gradient boosting regression on the training split.", """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42})
        model = GradientBoostingRegressor(random_state=${4:42})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("R2", r2_score(y_test, predictions))
        """, ["regression"], "implementations/08_Advanced_ML/boosting.py"),
    snippet("ml-regmetrics", "Regression metrics", "Print MAE, MSE, RMSE, R2, and adjusted R2.", """
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mse = mean_squared_error(${1:y_test}, ${2:predictions})
        r2 = r2_score(${1:y_test}, ${2:predictions})
        n = len(${1:y_test})
        p = ${3:X_test}.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        print("MAE", mean_absolute_error(${1:y_test}, ${2:predictions}))
        print("MSE", mse)
        print("RMSE", np.sqrt(mse))
        print("R2", r2)
        print("Adjusted R2", adjusted_r2)
        """, ["regression", "evaluate"], f"{E}/regression_metrics.py"),
    snippet("ml-regpipeline", "Regression pipeline", "Scale features and fit linear regression in one pipeline.", """
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
        pipeline.fit(${1:X_train}, ${2:y_train})
        predictions = pipeline.predict(${3:X_test})
        """, ["regression", "pipeline"], f"{R}/simple_linear_regression.py"),
    snippet("ml-logisticreg", "Logistic regression", "Fit logistic regression on a stratified split.", """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            ${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42}, stratify=${2:y}
        )
        model = LogisticRegression(max_iter=${5:1000}, random_state=${4:42})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("Accuracy", accuracy_score(y_test, predictions))
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))
        """, ["classification"], f"{C}/logistic_regression.py"),
    snippet("ml-logisticregmulti", "Multiclass logistic regression", "Fit logistic regression for more than two classes.", """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            ${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42}, stratify=${2:y}
        )
        model = LogisticRegression(max_iter=${5:1000}, random_state=${4:42})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        """, ["classification"], f"{C}/logistic_regression.py"),
    snippet("ml-knn", "KNN classifier", "Fit a K-nearest neighbors classifier on a stratified split.", """
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = train_test_split(
            ${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42}, stratify=${2:y}
        )
        model = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=${5:5})),
        ])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("Accuracy", accuracy_score(y_test, predictions))
        """, ["classification"], f"{C}/knn.py", extra_aliases=["ml-knnclassifier"]),
    snippet("ml-knntuner", "Tune KNN neighbors", "Search neighbor counts inside a scaled KNN pipeline.", """
        from sklearn.model_selection import GridSearchCV
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ])
        search = GridSearchCV(pipeline, {"knn__n_neighbors": [${1:3}, ${2:5}, ${3:7}, ${4:9}]}, cv=${5:5}, scoring="accuracy")
        search.fit(${6:X_train}, ${7:y_train})
        print(search.best_params_, search.best_score_)
        """, ["classification", "tune"], f"{C}/knn.py"),
    snippet("ml-naivebayes", "Gaussian naive Bayes", "Fit Gaussian naive Bayes on a stratified split.", """
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB

        X_train, X_test, y_train, y_test = train_test_split(
            ${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42}, stratify=${2:y}
        )
        model = GaussianNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("Accuracy", accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))
        """, ["classification"], f"{C}/naive_bayes.py"),
    snippet("ml-gaussiannb", "GaussianNB", "Fit GaussianNB on the training split.", """
        from sklearn.naive_bayes import GaussianNB

        model = GaussianNB()
        model.fit(${1:X_train}, ${2:y_train})
        predictions = model.predict(${3:X_test})
        """, ["classification"], f"{C}/naive_bayes.py"),
    snippet("ml-multinomialnb", "Multinomial naive Bayes", "Fit MultinomialNB on non-negative features such as counts.", """
        from sklearn.naive_bayes import MultinomialNB

        model = MultinomialNB()
        model.fit(${1:X_train}, ${2:y_train})
        predictions = model.predict(${3:X_test})
        """, ["classification"], f"{C}/naive_bayes.py"),
    snippet("ml-bernoullinb", "Bernoulli naive Bayes", "Fit BernoulliNB on binary features.", """
        from sklearn.naive_bayes import BernoulliNB

        model = BernoulliNB()
        model.fit(${1:X_train}, ${2:y_train})
        predictions = model.predict(${3:X_test})
        """, ["classification"], f"{C}/naive_bayes.py"),
    snippet("ml-decisiontreeclassifier", "Decision tree classifier", "Fit a decision tree on a stratified split.", """
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            ${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42}, stratify=${2:y}
        )
        model = DecisionTreeClassifier(max_depth=${5:4}, random_state=${4:42})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("Accuracy", accuracy_score(y_test, predictions))
        """, ["classification"], f"{C}/decision_tree.py"),
    snippet("ml-decisiontree", "Decision tree", "Fit a classification tree on the training split.", """
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(max_depth=${1:4}, random_state=${2:42})
        model.fit(${3:X_train}, ${4:y_train})
        predictions = model.predict(${5:X_test})
        """, ["classification"], f"{C}/decision_tree.py"),
    snippet("ml-randomforestclassifier", "Random forest classifier", "Fit a random forest on a stratified split.", """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            ${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42}, stratify=${2:y}
        )
        model = RandomForestClassifier(n_estimators=${5:100}, random_state=${4:42})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("Accuracy", accuracy_score(y_test, predictions))
        """, ["classification"], f"{C}/random_forest.py"),
    snippet("ml-randomforest", "Random forest", "Fit a random forest classifier on the training split.", """
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=${1:100}, random_state=${2:42})
        model.fit(${3:X_train}, ${4:y_train})
        predictions = model.predict(${5:X_test})
        """, ["classification"], f"{C}/random_forest.py"),
    snippet("ml-svmclassifier", "SVM classifier", "Fit an RBF SVM on features scaled inside a pipeline.", """
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        X_train, X_test, y_train, y_test = train_test_split(
            ${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42}, stratify=${2:y}
        )
        model = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=${5:"rbf"}, C=${6:1.0}, random_state=${4:42})),
        ])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("Accuracy", accuracy_score(y_test, predictions))
        """, ["classification"], f"{C}/svm.py"),
    snippet("ml-svm", "SVC", "Fit a support vector classifier on the training split.", """
        from sklearn.svm import SVC

        model = SVC(kernel=${1:"rbf"}, C=${2:1.0}, random_state=${3:42})
        model.fit(${4:X_train}, ${5:y_train})
        predictions = model.predict(${6:X_test})
        """, ["classification"], f"{C}/svm.py"),
    snippet("ml-linearsvc", "Linear SVC", "Fit a linear support vector classifier.", """
        from sklearn.svm import LinearSVC

        model = LinearSVC(C=${1:1.0}, random_state=${2:42}, max_iter=${3:5000})
        model.fit(${4:X_train}, ${5:y_train})
        predictions = model.predict(${6:X_test})
        """, ["classification"], f"{C}/svm.py"),
    snippet("ml-gradientboostingclassifier", "Gradient boosting classifier", "Fit gradient boosting on a stratified split.", """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            ${1:X}, ${2:y}, test_size=${3:0.2}, random_state=${4:42}, stratify=${2:y}
        )
        model = GradientBoostingClassifier(random_state=${4:42})
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("Accuracy", accuracy_score(y_test, predictions))
        """, ["classification"], f"{C}/gradient_boosting.py"),
    snippet("ml-classmetrics", "Classification metrics", "Print accuracy, precision, recall, and F1.", """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        print("Accuracy", accuracy_score(${1:y_test}, ${2:predictions}))
        print("Precision", precision_score(${1:y_test}, ${2:predictions}, average=${3:"weighted"}, zero_division=0))
        print("Recall", recall_score(${1:y_test}, ${2:predictions}, average=${3:"weighted"}, zero_division=0))
        print("F1", f1_score(${1:y_test}, ${2:predictions}, average=${3:"weighted"}, zero_division=0))
        """, ["classification", "evaluate"], f"{E}/classification_metrics.py"),
    snippet("ml-classificationreport", "Classification report", "Print a classification report.", """
        from sklearn.metrics import classification_report

        print(classification_report(${1:y_test}, ${2:predictions}, zero_division=0))
        """, ["classification", "evaluate"], f"{E}/classification_metrics.py"),
    snippet("ml-roc", "ROC curve", "Plot a binary ROC curve from predicted probabilities.", """
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay

        RocCurveDisplay.from_predictions(${1:y_test}, ${2:probabilities})
        plt.title("ROC curve")
        plt.show()
        """, ["classification", "evaluate"], f"{E}/classification_metrics.py"),
    snippet("ml-classpipeline", "Classification pipeline", "Scale features and fit logistic regression in one pipeline.", """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=${1:1000})),
        ])
        pipeline.fit(${2:X_train}, ${3:y_train})
        predictions = pipeline.predict(${4:X_test})
        """, ["classification", "pipeline"], f"{C}/logistic_regression.py"),
    snippet("ml-kmeans", "K-Means", "Scale features and fit K-Means.", """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(${1:X})
        model = KMeans(n_clusters=${2:3}, n_init=${3:10}, random_state=${4:42})
        labels = model.fit_predict(X_scaled)
        """, ["cluster"], f"{K}/kmeans.py"),
    snippet("ml-hierarchical", "Hierarchical clustering", "Scale features and fit agglomerative clustering.", """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(${1:X})
        model = AgglomerativeClustering(n_clusters=${2:3}, linkage=${3:"ward"})
        labels = model.fit_predict(X_scaled)
        """, ["cluster"], f"{K}/hierarchical.py"),
    snippet("ml-dbscan", "DBSCAN", "Scale features and fit DBSCAN.", """
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(${1:X})
        model = DBSCAN(eps=${2:0.5}, min_samples=${3:5})
        labels = model.fit_predict(X_scaled)
        """, ["cluster"], f"{K}/dbscan.py"),
    snippet("ml-elbow", "Elbow method", "Plot K-Means inertia for a range of cluster counts.", """
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(${1:X})
        inertias = []
        k_values = range(${2:1}, ${3:11})
        for k in k_values:
            model = KMeans(n_clusters=k, n_init=10, random_state=${4:42})
            model.fit(X_scaled)
            inertias.append(model.inertia_)
        plt.plot(list(k_values), inertias, marker="o")
        plt.xlabel("Clusters")
        plt.ylabel("Inertia")
        plt.show()
        """, ["cluster"], f"{K}/elbow_method.py"),
    snippet("ml-silhouette", "Silhouette score", "Compare silhouette scores for several cluster counts.", """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(${1:X})
        for k in range(${2:2}, ${3:8}):
            labels = KMeans(n_clusters=k, n_init=10, random_state=${4:42}).fit_predict(X_scaled)
            print(k, silhouette_score(X_scaled, labels))
        """, ["cluster"], f"{K}/silhouette_analysis.py"),
    snippet("ml-dendrogram", "Dendrogram", "Draw a hierarchical clustering dendrogram.", """
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(${1:X})
        dendrogram(linkage(X_scaled, method=${2:"ward"}))
        plt.title("Dendrogram")
        plt.show()
        """, ["cluster"], f"{K}/hierarchical.py"),
    snippet("ml-clustervisual", "Plot clusters", "Scatter the first two scaled features colored by cluster label.", """
        import matplotlib.pyplot as plt

        plt.scatter(${1:X_scaled}[:, 0], ${1:X_scaled}[:, 1], c=${2:labels})
        plt.title("Clusters")
        plt.show()
        """, ["cluster"], f"{K}/visualization.py"),
    snippet("ml-clusterassign", "Cluster assignments", "Attach K-Means labels to a DataFrame.", """
        ${1:df}[${2:"cluster"}] = ${3:labels}
        """, ["cluster"], f"{K}/kmeans.py"),
    snippet("ml-clusterscale", "Scale before clustering", "Standardize a matrix before a distance-based clustering method.", """
        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(${1:X})
        """, ["cluster", "scale"], f"{K}/kmeans.py"),
    snippet("ml-mae", "MAE", "Mean absolute error.", """
        from sklearn.metrics import mean_absolute_error

        print(mean_absolute_error(${1:y_test}, ${2:predictions}))
        """, ["evaluate"], f"{E}/regression_metrics.py"),
    snippet("ml-mse", "MSE", "Mean squared error.", """
        from sklearn.metrics import mean_squared_error

        print(mean_squared_error(${1:y_test}, ${2:predictions}))
        """, ["evaluate"], f"{E}/regression_metrics.py"),
    snippet("ml-rmse", "RMSE", "Root mean squared error.", """
        from sklearn.metrics import mean_squared_error

        print(mean_squared_error(${1:y_test}, ${2:predictions}) ** 0.5)
        """, ["evaluate"], f"{E}/regression_metrics.py"),
    snippet("ml-r2", "R-squared", "Coefficient of determination.", """
        from sklearn.metrics import r2_score

        print(r2_score(${1:y_test}, ${2:predictions}))
        """, ["evaluate"], f"{E}/regression_metrics.py"),
    snippet("ml-adjustedr2", "Adjusted R-squared", "R-squared adjusted for the number of predictors.", """
        from sklearn.metrics import r2_score

        r2 = r2_score(${1:y_test}, ${2:predictions})
        n = len(${1:y_test})
        p = ${3:X_test}.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        print(adjusted_r2)
        """, ["evaluate"], f"{E}/regression_metrics.py"),
    snippet("ml-accuracy", "Accuracy", "Classification accuracy.", """
        from sklearn.metrics import accuracy_score

        print(accuracy_score(${1:y_test}, ${2:predictions}))
        """, ["evaluate"], f"{E}/classification_metrics.py"),
    snippet("ml-precisionrecallf1", "Precision recall F1", "Weighted precision, recall, and F1.", """
        from sklearn.metrics import f1_score, precision_score, recall_score

        print(precision_score(${1:y_test}, ${2:predictions}, average="weighted", zero_division=0))
        print(recall_score(${1:y_test}, ${2:predictions}, average="weighted", zero_division=0))
        print(f1_score(${1:y_test}, ${2:predictions}, average="weighted", zero_division=0))
        """, ["evaluate"], f"{E}/classification_metrics.py"),
    snippet("ml-confusionmatrix", "Confusion matrix", "Build and plot a confusion matrix.", """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        matrix = confusion_matrix(${1:y_test}, ${2:predictions})
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        """, ["evaluate"], f"{E}/confusion_matrix.py"),
    snippet("ml-rocauc", "ROC AUC", "Binary ROC AUC from positive-class probabilities.", """
        from sklearn.metrics import roc_auc_score

        print(roc_auc_score(${1:y_test}, ${2:probabilities}))
        """, ["evaluate"], f"{E}/classification_metrics.py"),
    snippet("ml-learningcurve", "Learning curve", "Compute a learning curve for an estimator or pipeline.", """
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, validation_scores = learning_curve(
            ${1:model}, ${2:X}, ${3:y}, cv=${4:5}, scoring=${5:"accuracy"}, n_jobs=1
        )
        """, ["evaluate"], f"{E}/model_diagnostics.py"),
    snippet("ml-gridsearch", "Grid search", "Search hyperparameters with cross-validation.", """
        from sklearn.model_selection import GridSearchCV

        param_grid = {"max_depth": [${1:2}, ${2:4}, ${3:6}]}
        search = GridSearchCV(${4:model}, param_grid=param_grid, cv=${5:5}, scoring=${6:"accuracy"})
        search.fit(${7:X_train}, ${8:y_train})
        print(search.best_params_, search.best_score_)
        """, ["tune"], f"{E}/hyperparameter_tuning.py"),
    snippet("ml-randomsearch", "Randomized search", "Sample hyperparameters with cross-validation.", """
        from sklearn.model_selection import RandomizedSearchCV

        param_distributions = {"max_depth": [${1:2}, ${2:4}, ${3:6}, ${4:8}]}
        search = RandomizedSearchCV(
            ${5:model},
            param_distributions=param_distributions,
            n_iter=${6:8},
            cv=${7:5},
            scoring=${8:"accuracy"},
            random_state=${9:42},
        )
        search.fit(${10:X_train}, ${11:y_train})
        print(search.best_params_, search.best_score_)
        """, ["tune"], f"{E}/hyperparameter_tuning.py"),
    snippet("ml-crossvalidate", "Cross-validate", "Score an estimator with stratified or plain K-fold.", """
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(${1:model}, ${2:X}, ${3:y}, cv=${4:5}, scoring=${5:"accuracy"})
        print(scores.mean(), scores.std())
        """, ["tune"], f"{E}/cross_validation.py"),
    snippet("ml-modelcompare", "Compare models", "Compare pipelines with the same cross-validation score.", """
        from sklearn.model_selection import cross_val_score

        for name, model in ${1:models}.items():
            scores = cross_val_score(model, ${2:X}, ${3:y}, cv=${4:5}, scoring=${5:"f1_weighted"})
            print(name, scores.mean())
        """, ["tune"], f"{E}/model_comparison.py"),
    snippet("ml-validationcurve", "Validation curve", "Plot train and validation scores across one parameter.", """
        from sklearn.model_selection import validation_curve

        train_scores, validation_scores = validation_curve(
            ${1:model}, ${2:X}, ${3:y}, param_name=${4:"max_depth"}, param_range=[${5:2}, ${6:4}, ${7:6}], cv=${8:5}
        )
        """, ["tune"], f"{E}/model_diagnostics.py"),
    snippet("ml-threshold", "Classification threshold", "Apply a custom probability threshold for the positive class.", """
        positive_probability = ${1:model}.predict_proba(${2:X_test})[:, 1]
        predictions = (positive_probability >= ${3:0.5}).astype(int)
        """, ["tune"], f"{E}/classification_metrics.py"),
    snippet("ml-joblibsave", "Save with joblib", "Save a fitted model with joblib.", """
        import joblib

        joblib.dump(${1:model}, ${2:"model.joblib"})
        """, ["save"], f"{E}/model_persistence.py"),
    snippet("ml-joblibload", "Load with joblib", "Load a model saved with joblib.", """
        import joblib

        model = joblib.load(${1:"model.joblib"})
        """, ["save"], f"{E}/model_persistence.py"),
    snippet("ml-picklesave", "Save with pickle", "Save a fitted model with pickle.", """
        import pickle

        with open(${1:"model.pkl"}, "wb") as model_file:
            pickle.dump(${2:model}, model_file)
        """, ["save"], f"{E}/model_persistence.py"),
    snippet("ml-pickleload", "Load with pickle", "Load a model saved with pickle.", """
        import pickle

        with open(${1:"model.pkl"}, "rb") as model_file:
            model = pickle.load(model_file)
        """, ["save"], f"{E}/model_persistence.py"),
]
