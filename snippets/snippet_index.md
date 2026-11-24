# Snippet index

212 snippets. Type `MLToolKit.ml_` after importing the package.

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

## clean

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_astype` | Cast selected columns to new dtypes. | `MLToolKit/implementations/01_Data_Preprocessing/inconsistent_data.py` |
| `ml_invalidvalues` | Flag values outside an allowed set. | `MLToolKit/implementations/01_Data_Preprocessing/inconsistent_data.py` |
| `ml_lowercase` | Strip and lowercase text columns. | `MLToolKit/implementations/01_Data_Preprocessing/inconsistent_data.py` |
| `ml_rangevalidation` | Flag numeric values outside a closed range. | `MLToolKit/implementations/01_Data_Preprocessing/inconsistent_data.py` |
| `ml_renamecolumns` | Rename columns with a mapping. | `MLToolKit/implementations/09_Utilities/pandas_utilities.py` |
| `ml_standardizecategory` | Replace inconsistent category labels. | `MLToolKit/implementations/01_Data_Preprocessing/inconsistent_data.py` |
| `ml_stripwhitespace` | Trim text columns. | `MLToolKit/implementations/01_Data_Preprocessing/inconsistent_data.py` |
| `ml_todatetime` | Parse a column into datetimes. | `MLToolKit/implementations/01_Data_Preprocessing/inconsistent_data.py` |
| `ml_tonumeric` | Convert a column to numbers and mark invalid entries as missing. | `MLToolKit/implementations/01_Data_Preprocessing/inconsistent_data.py` |
| `ml_unitconvert` | Multiply a column by a conversion factor. | `MLToolKit/implementations/01_Data_Preprocessing/inconsistent_data.py` |
| `ml_uppercase` | Strip and uppercase a text column. | `MLToolKit/implementations/01_Data_Preprocessing/inconsistent_data.py` |

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

## deep

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_kerasann` | Train a small Keras network. Requires tensorflow. | `MLToolKit/implementations/10_Deep_Learning/tensorflow/ann.py` |
| `ml_torchann` | Train a small dropout network. Requires torch. | `MLToolKit/implementations/10_Deep_Learning/pytorch/ann.py` |

## deploy

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_fastapi` | Serve a joblib model with FastAPI. Requires fastapi and pydantic. | `MLToolKit/implementations/13_Deployment_MLOps/fastapi_template.py` |
| `ml_mlflow` | Write parameters and metrics to a JSON file without a tracking server. | `MLToolKit/implementations/13_Deployment_MLOps/mlflow_tracking.py` |

## duplicate

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_dropduplicates` | Remove duplicate rows and keep the first copy. | `MLToolKit/implementations/01_Data_Preprocessing/duplicate_handling.py` |
| `ml_dropduplicateskeepfirst` | Drop duplicates and keep the first record. | `MLToolKit/implementations/01_Data_Preprocessing/duplicate_handling.py` |
| `ml_dropduplicateskeeplast` | Drop duplicates and keep the last record. | `MLToolKit/implementations/01_Data_Preprocessing/duplicate_handling.py` |
| `ml_dropduplicatessubset` | Remove rows that repeat the selected columns. | `MLToolKit/implementations/01_Data_Preprocessing/duplicate_handling.py` |
| `ml_duplicated` | Mark duplicate rows. | `MLToolKit/implementations/01_Data_Preprocessing/duplicate_handling.py` |

## eda

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_agg` | Group and aggregate several statistics. | `MLToolKit/implementations/02_EDA/descriptive_statistics.py` |
| `ml_boxplot` | Draw a boxplot. | `MLToolKit/implementations/02_EDA/visualization.py` |
| `ml_concat` | Stack frames by rows. | `MLToolKit/implementations/02_EDA/data_summary.py` |
| `ml_correlation` | Pearson correlation of numeric columns. | `MLToolKit/implementations/02_EDA/correlation.py` |
| `ml_corrheatmap` | Draw a correlation heatmap. | `MLToolKit/implementations/02_EDA/visualization.py` |
| `ml_countplot` | Draw a Seaborn count plot. | `MLToolKit/implementations/02_EDA/visualization.py` |
| `ml_crosstab` | Cross-tabulate two categorical columns. | `MLToolKit/implementations/02_EDA/categorical_analysis.py` |
| `ml_distplot` | Draw a histogram with a density curve. | `MLToolKit/implementations/02_EDA/visualization.py` |
| `ml_groupby` | Aggregate one column by another. | `MLToolKit/implementations/02_EDA/data_summary.py` |
| `ml_histogram` | Draw a histogram. | `MLToolKit/implementations/02_EDA/visualization.py` |
| `ml_join` | Join two frames by index. | `MLToolKit/implementations/02_EDA/data_summary.py` |
| `ml_merge` | Merge two frames on a key. | `MLToolKit/implementations/02_EDA/data_summary.py` |
| `ml_missingheatmap` | Show which cells are missing. | `MLToolKit/implementations/02_EDA/visualization.py` |
| `ml_pairplot` | Draw a Seaborn pair plot. | `MLToolKit/implementations/02_EDA/visualization.py` |
| `ml_pivot` | Build a pivot table. | `MLToolKit/implementations/02_EDA/data_summary.py` |
| `ml_scatter` | Draw a scatter plot. | `MLToolKit/implementations/02_EDA/visualization.py` |

## encode

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_frequencyencode` | Replace categories with training frequencies. | `MLToolKit/implementations/01_Data_Preprocessing/categorical_encoding.py` |
| `ml_getdummies` | Create dummies from training columns and align the test frame. | `MLToolKit/implementations/01_Data_Preprocessing/categorical_encoding.py` |
| `ml_labelencode` | Label-encode one column. Unseen test labels become -1. | `MLToolKit/implementations/01_Data_Preprocessing/categorical_encoding.py` |
| `ml_onehot` | One-hot encode categorical columns and ignore unknown test values. | `MLToolKit/implementations/01_Data_Preprocessing/categorical_encoding.py` |
| `ml_onehotignore` | Fit one-hot encoding on training categories only. | `MLToolKit/implementations/01_Data_Preprocessing/categorical_encoding.py` |
| `ml_ordinalencode` | Ordinal-encode columns and keep unknown test values as -1. | `MLToolKit/implementations/01_Data_Preprocessing/categorical_encoding.py` |

## evaluate

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_accuracy` | Classification accuracy. | `MLToolKit/implementations/06_Model_Evaluation/classification_metrics.py` |
| `ml_adjustedr2` | R-squared adjusted for the number of predictors. | `MLToolKit/implementations/06_Model_Evaluation/regression_metrics.py` |
| `ml_confusionmatrix` | Build and plot a confusion matrix. | `MLToolKit/implementations/06_Model_Evaluation/confusion_matrix.py` |
| `ml_learningcurve` | Compute a learning curve for an estimator or pipeline. | `MLToolKit/implementations/06_Model_Evaluation/model_diagnostics.py` |
| `ml_mae` | Mean absolute error. | `MLToolKit/implementations/06_Model_Evaluation/regression_metrics.py` |
| `ml_mse` | Mean squared error. | `MLToolKit/implementations/06_Model_Evaluation/regression_metrics.py` |
| `ml_precisionrecallf1` | Weighted precision, recall, and F1. | `MLToolKit/implementations/06_Model_Evaluation/classification_metrics.py` |
| `ml_r2` | Coefficient of determination. | `MLToolKit/implementations/06_Model_Evaluation/regression_metrics.py` |
| `ml_rmse` | Root mean squared error. | `MLToolKit/implementations/06_Model_Evaluation/regression_metrics.py` |
| `ml_rocauc` | Binary ROC AUC from positive-class probabilities. | `MLToolKit/implementations/06_Model_Evaluation/classification_metrics.py` |

## impute

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_imputebackwardfill` | Backward-fill each split separately. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |
| `ml_imputeconstant` | Fill gaps with a chosen constant. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |
| `ml_imputeforwardfill` | Forward-fill each split without using test rows in training. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |
| `ml_imputeknn` | Fit KNNImputer on the training matrix. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |
| `ml_imputemean` | Fill numeric gaps with the training mean. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |
| `ml_imputemedian` | Fill numeric gaps with the training median. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |
| `ml_imputemode` | Fill gaps with the training mode. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |
| `ml_imputepipeline` | Median imputation stays inside the model pipeline. | `MLToolKit/implementations/01_Data_Preprocessing/preprocessing_pipeline.py` |
| `ml_imputesimple` | Fit SimpleImputer on the training matrix. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |
| `ml_missingindicator` | Add missingness flags fit on the training matrix. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |

## inspect

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_checkmissing` | Count and percentage of missing values. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |
| `ml_columns` | List the column names. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |
| `ml_describe` | Summarize numeric and categorical columns. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |
| `ml_dtypes` | Show the dtype of every column. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |
| `ml_head` | Show the first rows. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |
| `ml_info` | Print column dtypes and non-null counts. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |
| `ml_isna` | Boolean missing-value mask using isna. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |
| `ml_isnull` | Boolean missing-value mask using isnull. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |
| `ml_memoryusage` | Report deep memory usage by column. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |
| `ml_nunique` | Count unique values in every column. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |
| `ml_sample` | Draw a reproducible sample of rows. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |
| `ml_shape` | Read the row and column counts. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |
| `ml_tail` | Show the last rows. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |
| `ml_valuecounts` | Count categories in one column. | `MLToolKit/implementations/01_Data_Preprocessing/data_inspection.py` |

## load

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_loadclipboard` | Read a table that was copied to the clipboard. | `MLToolKit/implementations/01_Data_Preprocessing/data_loading.py` |
| `ml_loadcsv` | Read a CSV file with Pandas. | `MLToolKit/implementations/01_Data_Preprocessing/data_loading.py` |
| `ml_loadcsvoptions` | Read a CSV with a custom separator and encoding. | `MLToolKit/implementations/01_Data_Preprocessing/data_loading.py` |
| `ml_loadcsvpath` | Read a CSV from a relative or absolute path. | `MLToolKit/implementations/01_Data_Preprocessing/data_loading.py` |
| `ml_loaddataset` | Choose a CSV, Excel, JSON, or NumPy reader from the suffix. | `MLToolKit/implementations/01_Data_Preprocessing/data_loading.py` |
| `ml_loadexcel` | Read an Excel worksheet with Pandas. | `MLToolKit/implementations/01_Data_Preprocessing/data_loading.py` |
| `ml_loadjson` | Read a JSON file into a DataFrame. | `MLToolKit/implementations/01_Data_Preprocessing/data_loading.py` |
| `ml_loadmultiple` | Read every CSV in a folder into a dictionary of frames. | `MLToolKit/implementations/01_Data_Preprocessing/data_loading.py` |
| `ml_loadnumpy` | Load a .npy array saved with numpy.save. | `MLToolKit/implementations/01_Data_Preprocessing/data_loading.py` |
| `ml_loadnumpytext` | Load a delimited numeric file with NumPy. | `MLToolKit/implementations/01_Data_Preprocessing/data_loading.py` |
| `ml_loadsql` | Read a SQL query into a DataFrame. | `MLToolKit/implementations/01_Data_Preprocessing/data_loading.py` |
| `ml_readdataset` | Read a CSV dataset and keep a copy of the raw frame. | `MLToolKit/implementations/01_Data_Preprocessing/data_loading.py` |

## missing

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_dropmissing` | Drop rows that contain missing values. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |
| `ml_fillna` | Fill missing values with a constant. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |
| `ml_missingcount` | Count missing values in each column. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |
| `ml_missingpercent` | Missing values as a percentage of rows. | `MLToolKit/implementations/01_Data_Preprocessing/missing_values.py` |

## nlp

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_bow` | Fit word counts on training documents. | `MLToolKit/implementations/11_NLP/bag_of_words.py` |
| `ml_textclassify` | Classify text with TF-IDF and logistic regression in one pipeline. | `MLToolKit/implementations/11_NLP/text_classification.py` |
| `ml_textclean` | Lowercase text and remove URLs and punctuation. | `MLToolKit/implementations/11_NLP/text_preprocessing.py` |
| `ml_tfidf` | Fit TF-IDF on training documents and transform the test documents. | `MLToolKit/implementations/11_NLP/tfidf.py` |

## outlier

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_capoutliers` | Clip both splits to fences learned from training data. | `MLToolKit/implementations/01_Data_Preprocessing/outlier_handling.py` |
| `ml_detectoutliers` | Mark training rows outside the training fences. | `MLToolKit/implementations/01_Data_Preprocessing/outlier_handling.py` |
| `ml_iqr` | Compute outlier fences from the training split. | `MLToolKit/implementations/01_Data_Preprocessing/outlier_handling.py` |
| `ml_outlierboxplot` | Draw a boxplot for one numeric column. | `MLToolKit/implementations/02_EDA/visualization.py` |
| `ml_removeoutliers` | Drop training rows outside the training fences. | `MLToolKit/implementations/01_Data_Preprocessing/outlier_handling.py` |
| `ml_winsorize` | Winsorize one series with SciPy. | `MLToolKit/implementations/01_Data_Preprocessing/outlier_handling.py` |
| `ml_zscore` | Compute z-scores from the training mean and standard deviation. | `MLToolKit/implementations/01_Data_Preprocessing/outlier_handling.py` |

## pipeline

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_columntransformer` | Impute and encode numeric and categorical columns separately. | `MLToolKit/implementations/01_Data_Preprocessing/preprocessing_pipeline.py` |
| `ml_fullpipeline` | Fit imputation, encoding, scaling, and the model on training data only. | `MLToolKit/implementations/01_Data_Preprocessing/preprocessing_pipeline.py` |
| `ml_pipeline` | Chain preprocessing and a model so every fold fits on training rows only. | `MLToolKit/implementations/01_Data_Preprocessing/preprocessing_pipeline.py` |
| `ml_preprocesspipeline` | Fit a ColumnTransformer on training features and transform the test features. | `MLToolKit/implementations/01_Data_Preprocessing/preprocessing_pipeline.py` |

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

## save

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_joblibload` | Load a model saved with joblib. | `MLToolKit/implementations/06_Model_Evaluation/model_persistence.py` |
| `ml_joblibsave` | Save a fitted model with joblib. | `MLToolKit/implementations/06_Model_Evaluation/model_persistence.py` |
| `ml_pickleload` | Load a model saved with pickle. | `MLToolKit/implementations/06_Model_Evaluation/model_persistence.py` |
| `ml_picklesave` | Save a fitted model with pickle. | `MLToolKit/implementations/06_Model_Evaluation/model_persistence.py` |

## scale

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_columnscale` | Standardize chosen numeric columns from the training frame. | `MLToolKit/implementations/01_Data_Preprocessing/feature_scaling.py` |
| `ml_maxabsscale` | Scale by the training maximum absolute value. | `MLToolKit/implementations/01_Data_Preprocessing/feature_scaling.py` |
| `ml_minmaxscale` | Scale features to a range using training limits. | `MLToolKit/implementations/01_Data_Preprocessing/feature_scaling.py` |
| `ml_normalize` | Scale each row to unit norm. This does not use column statistics. | `MLToolKit/implementations/01_Data_Preprocessing/feature_scaling.py` |
| `ml_robustscale` | Scale with the training median and interquartile range. | `MLToolKit/implementations/01_Data_Preprocessing/feature_scaling.py` |
| `ml_scalepipeline` | Keep scaling inside the estimator pipeline. | `MLToolKit/implementations/01_Data_Preprocessing/preprocessing_pipeline.py` |
| `ml_scaletraintest` | Fit a scaler on training data and transform both splits. | `MLToolKit/implementations/01_Data_Preprocessing/feature_scaling.py` |
| `ml_standardscale` | Standardize features using the training mean and standard deviation. | `MLToolKit/implementations/01_Data_Preprocessing/feature_scaling.py` |

## select

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_anova` | Select K features with the ANOVA F-test. | `MLToolKit/implementations/01_Data_Preprocessing/feature_selection.py` |
| `ml_chi2` | Select K non-negative features with chi-square scores. | `MLToolKit/implementations/01_Data_Preprocessing/feature_selection.py` |
| `ml_featureimportance` | Rank features with a random forest fit on the training split. | `MLToolKit/implementations/01_Data_Preprocessing/feature_selection.py` |
| `ml_mutualinfo` | Select K features with mutual information. | `MLToolKit/implementations/01_Data_Preprocessing/feature_selection.py` |
| `ml_rfe` | Recursively eliminate features on the training split. | `MLToolKit/implementations/01_Data_Preprocessing/feature_selection.py` |
| `ml_selectkbest` | Keep the K highest scoring training features. | `MLToolKit/implementations/01_Data_Preprocessing/feature_selection.py` |
| `ml_variancethreshold` | Drop low-variance features using the training matrix. | `MLToolKit/implementations/01_Data_Preprocessing/feature_selection.py` |

## series

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_arima` | Fit an ARIMA model and forecast the next steps. | `MLToolKit/implementations/12_Time_Series/forecasting.py` |
| `ml_lagfeatures` | Add lagged copies of a series. | `MLToolKit/implementations/12_Time_Series/feature_extraction.py` |
| `ml_resample` | Resample a datetime-indexed series. | `MLToolKit/implementations/12_Time_Series/temporal_preprocessing.py` |
| `ml_rolling` | Rolling mean of past values, excluding the current row. | `MLToolKit/implementations/12_Time_Series/temporal_preprocessing.py` |

## split

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_kfold` | Build shuffled K-fold indices. | `MLToolKit/implementations/01_Data_Preprocessing/train_test_split.py` |
| `ml_randomseed` | Seed Python and NumPy before a practical. | `MLToolKit/implementations/09_Utilities/random_seed.py` |
| `ml_splitxy` | Separate the target column from the predictors. | `MLToolKit/implementations/01_Data_Preprocessing/train_test_split.py` |
| `ml_stratifiedkfold` | Build stratified K-fold indices. | `MLToolKit/implementations/01_Data_Preprocessing/train_test_split.py` |
| `ml_stratifiedsplit` | Preserve class proportions in the test split. | `MLToolKit/implementations/01_Data_Preprocessing/train_test_split.py` |
| `ml_timeseriessplit` | Expand the training prefix for each later test block. | `MLToolKit/implementations/01_Data_Preprocessing/train_test_split.py` |
| `ml_traintestsplit` | Split features and target with a fixed random state. | `MLToolKit/implementations/01_Data_Preprocessing/train_test_split.py` |
| `ml_traintestval` | Hold out test data before splitting a validation set. | `MLToolKit/implementations/01_Data_Preprocessing/train_test_split.py` |

## transform

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_binning` | Bin a numeric column with explicit edges. | `MLToolKit/implementations/01_Data_Preprocessing/numerical_transformation.py` |
| `ml_boxcox` | Fit a Box-Cox transformer on positive training data. | `MLToolKit/implementations/01_Data_Preprocessing/numerical_transformation.py` |
| `ml_datetimeparts` | Add year, month, day, and weekday columns. | `MLToolKit/implementations/07_Feature_Engineering/feature_creation.py` |
| `ml_kbins` | Learn bin edges on the training data. | `MLToolKit/implementations/01_Data_Preprocessing/numerical_transformation.py` |
| `ml_logtransform` | Apply log1p to a non-negative column. | `MLToolKit/implementations/01_Data_Preprocessing/numerical_transformation.py` |
| `ml_pca` | Scale and project onto principal components fit on the training split. | `MLToolKit/implementations/07_Feature_Engineering/pca.py` |
| `ml_polynomialfeatures` | Expand polynomial features inside a pipeline fit on training data. | `MLToolKit/implementations/07_Feature_Engineering/polynomial_features.py` |
| `ml_powertransform` | Fit a Yeo-Johnson transformer on the training columns. | `MLToolKit/implementations/01_Data_Preprocessing/numerical_transformation.py` |
| `ml_quantiletransform` | Fit a quantile transformer on the training columns. | `MLToolKit/implementations/01_Data_Preprocessing/numerical_transformation.py` |
| `ml_ratiofeature` | Divide one numeric column by another. | `MLToolKit/implementations/07_Feature_Engineering/feature_creation.py` |
| `ml_sqrttransform` | Apply a square root to a non-negative column. | `MLToolKit/implementations/01_Data_Preprocessing/numerical_transformation.py` |

## tune

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_crossvalidate` | Score an estimator with stratified or plain K-fold. | `MLToolKit/implementations/06_Model_Evaluation/cross_validation.py` |
| `ml_gridsearch` | Search hyperparameters with cross-validation. | `MLToolKit/implementations/06_Model_Evaluation/hyperparameter_tuning.py` |
| `ml_modelcompare` | Compare pipelines with the same cross-validation score. | `MLToolKit/implementations/06_Model_Evaluation/model_comparison.py` |
| `ml_randomsearch` | Sample hyperparameters with cross-validation. | `MLToolKit/implementations/06_Model_Evaluation/hyperparameter_tuning.py` |
| `ml_threshold` | Apply a custom probability threshold for the positive class. | `MLToolKit/implementations/06_Model_Evaluation/classification_metrics.py` |
| `ml_validationcurve` | Plot train and validation scores across one parameter. | `MLToolKit/implementations/06_Model_Evaluation/model_diagnostics.py` |

## utility

| Trigger | Description | Implementation |
| --- | --- | --- |
| `ml_setseed` | Set the random, NumPy, and PyTorch seeds when PyTorch is installed. | `MLToolKit/implementations/09_Utilities/random_seed.py` |
| `ml_syntheticclass` | Build a classification table with scikit-learn. | `MLToolKit/implementations/09_Utilities/synthetic_datasets.py` |
| `ml_syntheticcluster` | Build a blob cluster table. | `MLToolKit/implementations/09_Utilities/synthetic_datasets.py` |
| `ml_syntheticreg` | Build a regression table with scikit-learn. | `MLToolKit/implementations/09_Utilities/synthetic_datasets.py` |

