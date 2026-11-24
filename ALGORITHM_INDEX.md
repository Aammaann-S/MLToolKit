# Algorithm index

Algorithms inserted by the MLToolKit extension or called from the package modules.

## regression

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_decisiontreereg` | Fit a decision tree regressor on the training split. | `MLToolKit/implementations/03_Regression/decision_tree_regressor.py` |
| `ml_elasticnet` | Fit Elastic Net on the training split. | `MLToolKit/implementations/03_Regression/elasticnet_regression.py` |
| `ml_gradientboostingreg` | Fit gradient boosting regression on the training split. | `MLToolKit/implementations/08_Advanced_ML/boosting.py` |
| `ml_lassoreg` | Fit lasso regression on the training split. | `MLToolKit/implementations/03_Regression/lasso_regression.py` |
| `ml_linearreg` | Fit one-feature linear regression after splitting. | `MLToolKit/implementations/03_Regression/simple_linear_regression.py` |
| `ml_multilinearreg` | Fit linear regression with several predictors. | `MLToolKit/implementations/03_Regression/multiple_linear_regression.py` |
| `ml_polynomialreg` | Expand polynomial features inside a pipeline fit on the training split. | `MLToolKit/implementations/03_Regression/polynomial_regression.py` |
| `ml_randomforestreg` | Fit a random forest regressor on the training split. | `MLToolKit/implementations/03_Regression/random_forest_regressor.py` |
| `ml_regmetrics` | Print MAE, MSE, RMSE, R2, and adjusted R2. | `MLToolKit/implementations/06_Model_Evaluation/regression_metrics.py` |
| `ml_regpipeline` | Scale features and fit linear regression in one pipeline. | `MLToolKit/implementations/03_Regression/simple_linear_regression.py` |
| `ml_ridgereg` | Fit ridge regression on the training split. | `MLToolKit/implementations/03_Regression/ridge_regression.py` |
| `ml_svr` | Fit SVR on features scaled inside a pipeline. | `MLToolKit/implementations/03_Regression/regression_evaluation.py` |

## classification

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_bernoullinb` | Fit BernoulliNB on binary features. | `MLToolKit/implementations/04_Classification/naive_bayes.py` |
| `ml_classificationreport` | Print a classification report. | `MLToolKit/implementations/06_Model_Evaluation/classification_metrics.py` |
| `ml_classmetrics` | Print accuracy, precision, recall, and F1. | `MLToolKit/implementations/06_Model_Evaluation/classification_metrics.py` |
| `ml_classpipeline` | Scale features and fit logistic regression in one pipeline. | `MLToolKit/implementations/04_Classification/logistic_regression.py` |
| `ml_decisiontree` | Fit a classification tree on the training split. | `MLToolKit/implementations/04_Classification/decision_tree.py` |
| `ml_decisiontreeclassifier` | Fit a decision tree on a stratified split. | `MLToolKit/implementations/04_Classification/decision_tree.py` |
| `ml_gaussiannb` | Fit GaussianNB on the training split. | `MLToolKit/implementations/04_Classification/naive_bayes.py` |
| `ml_gradientboostingclassifier` | Fit gradient boosting on a stratified split. | `MLToolKit/implementations/04_Classification/gradient_boosting.py` |
| `ml_knn` | Fit a K-nearest neighbors classifier on a stratified split. | `MLToolKit/implementations/04_Classification/knn.py` |
| `ml_knntuner` | Search neighbor counts inside a scaled KNN pipeline. | `MLToolKit/implementations/04_Classification/knn.py` |
| `ml_linearsvc` | Fit a linear support vector classifier. | `MLToolKit/implementations/04_Classification/svm.py` |
| `ml_logisticreg` | Fit logistic regression on a stratified split. | `MLToolKit/implementations/04_Classification/logistic_regression.py` |
| `ml_logisticregmulti` | Fit logistic regression for more than two classes. | `MLToolKit/implementations/04_Classification/logistic_regression.py` |
| `ml_multinomialnb` | Fit MultinomialNB on non-negative features such as counts. | `MLToolKit/implementations/04_Classification/naive_bayes.py` |
| `ml_naivebayes` | Fit Gaussian naive Bayes on a stratified split. | `MLToolKit/implementations/04_Classification/naive_bayes.py` |
| `ml_randomforest` | Fit a random forest classifier on the training split. | `MLToolKit/implementations/04_Classification/random_forest.py` |
| `ml_randomforestclassifier` | Fit a random forest on a stratified split. | `MLToolKit/implementations/04_Classification/random_forest.py` |
| `ml_roc` | Plot a binary ROC curve from predicted probabilities. | `MLToolKit/implementations/06_Model_Evaluation/classification_metrics.py` |
| `ml_svm` | Fit a support vector classifier on the training split. | `MLToolKit/implementations/04_Classification/svm.py` |
| `ml_svmclassifier` | Fit an RBF SVM on features scaled inside a pipeline. | `MLToolKit/implementations/04_Classification/svm.py` |

## cluster

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_clusterassign` | Attach K-Means labels to a DataFrame. | `MLToolKit/implementations/05_Clustering/kmeans.py` |
| `ml_clusterscale` | Standardize a matrix before a distance-based clustering method. | `MLToolKit/implementations/05_Clustering/kmeans.py` |
| `ml_clustervisual` | Scatter the first two scaled features colored by cluster label. | `MLToolKit/implementations/05_Clustering/visualization.py` |
| `ml_dbscan` | Scale features and fit DBSCAN. | `MLToolKit/implementations/05_Clustering/dbscan.py` |
| `ml_dendrogram` | Draw a hierarchical clustering dendrogram. | `MLToolKit/implementations/05_Clustering/hierarchical.py` |
| `ml_elbow` | Plot K-Means inertia for a range of cluster counts. | `MLToolKit/implementations/05_Clustering/elbow_method.py` |
| `ml_hierarchical` | Scale features and fit agglomerative clustering. | `MLToolKit/implementations/05_Clustering/hierarchical.py` |
| `ml_kmeans` | Scale features and fit K-Means. | `MLToolKit/implementations/05_Clustering/kmeans.py` |
| `ml_silhouette` | Compare silhouette scores for several cluster counts. | `MLToolKit/implementations/05_Clustering/silhouette_analysis.py` |

## advanced

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_adaboost` | Fit an AdaBoost classifier on the training split. | `MLToolKit/implementations/08_Advanced_ML/ensemble_learning.py` |
| `ml_catboost` | Fit CatBoost on the training split. Requires catboost. | `MLToolKit/implementations/08_Advanced_ML/boosting.py` |
| `ml_classweight` | Fit logistic regression with balanced class weights. | `MLToolKit/implementations/08_Advanced_ML/imbalanced.py` |
| `ml_extratrees` | Fit an extra-trees classifier on the training split. | `MLToolKit/implementations/08_Advanced_ML/ensemble_learning.py` |
| `ml_gmm` | Fit a Gaussian mixture on scaled features. | `MLToolKit/implementations/08_Advanced_ML/advanced_clustering.py` |
| `ml_isolationforest` | Flag anomalies with Isolation Forest. | `MLToolKit/implementations/08_Advanced_ML/anomaly_detection.py` |
| `ml_lightgbm` | Fit LightGBM on the training split. Requires lightgbm. | `MLToolKit/implementations/08_Advanced_ML/boosting.py` |
| `ml_lof` | Flag anomalies with local outlier factor. | `MLToolKit/implementations/08_Advanced_ML/anomaly_detection.py` |
| `ml_smote` | Resample the training split with SMOTE. Requires imbalanced-learn. | `MLToolKit/implementations/08_Advanced_ML/imbalanced.py` |
| `ml_spectral` | Fit spectral clustering on scaled features. | `MLToolKit/implementations/08_Advanced_ML/advanced_clustering.py` |
| `ml_voting` | Combine logistic regression, a tree, and KNN. | `MLToolKit/implementations/08_Advanced_ML/ensemble_learning.py` |
| `ml_xgboost` | Fit XGBoost on a stratified split. Requires xgboost. | `MLToolKit/implementations/08_Advanced_ML/boosting.py` |

## deep

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_kerasann` | Train a small Keras network. Requires tensorflow. | `MLToolKit/implementations/10_Deep_Learning/tensorflow/ann.py` |
| `ml_torchann` | Train a small dropout network. Requires torch. | `MLToolKit/implementations/10_Deep_Learning/pytorch/ann.py` |

## nlp

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_bow` | Fit word counts on training documents. | `MLToolKit/implementations/11_NLP/bag_of_words.py` |
| `ml_textclassify` | Classify text with TF-IDF and logistic regression in one pipeline. | `MLToolKit/implementations/11_NLP/text_classification.py` |
| `ml_textclean` | Lowercase text and remove URLs and punctuation. | `MLToolKit/implementations/11_NLP/text_preprocessing.py` |
| `ml_tfidf` | Fit TF-IDF on training documents and transform the test documents. | `MLToolKit/implementations/11_NLP/tfidf.py` |

## series

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_arima` | Fit an ARIMA model and forecast the next steps. | `MLToolKit/implementations/12_Time_Series/forecasting.py` |
| `ml_lagfeatures` | Add lagged copies of a series. | `MLToolKit/implementations/12_Time_Series/feature_extraction.py` |
| `ml_resample` | Resample a datetime-indexed series. | `MLToolKit/implementations/12_Time_Series/temporal_preprocessing.py` |
| `ml_rolling` | Rolling mean of past values, excluding the current row. | `MLToolKit/implementations/12_Time_Series/temporal_preprocessing.py` |

