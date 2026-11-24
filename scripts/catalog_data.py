"""Data loading, inspection, missing values, duplicates, and cleaning snippets."""

from catalog_common import snippet

P = "implementations/01_Data_Preprocessing"

SNIPPETS = [
    snippet("ml-loadcsv", "Load CSV", "Read a CSV file with Pandas.", """
        import pandas as pd

        df = pd.read_csv(${1:"data.csv"})
        """, ["load"], f"{P}/data_loading.py"),
    snippet("ml-loadcsvpath", "Load CSV from a path", "Read a CSV from a relative or absolute path.", """
        import pandas as pd

        df = pd.read_csv(${1:"data/dataset.csv"})
        """, ["load"], f"{P}/data_loading.py"),
    snippet("ml-loadcsvoptions", "Load CSV with options", "Read a CSV with a custom separator and encoding.", """
        import pandas as pd

        df = pd.read_csv(
            ${1:"data.csv"},
            sep=${2:","},
            encoding=${3:"utf-8"},
            header=${4:0},
        )
        """, ["load"], f"{P}/data_loading.py"),
    snippet("ml-loadexcel", "Load Excel", "Read an Excel worksheet with Pandas.", """
        import pandas as pd

        df = pd.read_excel(${1:"data.xlsx"}, sheet_name=${2:0})
        """, ["load"], f"{P}/data_loading.py"),
    snippet("ml-loadjson", "Load JSON", "Read a JSON file into a DataFrame.", """
        import pandas as pd

        df = pd.read_json(${1:"data.json"})
        """, ["load"], f"{P}/data_loading.py"),
    snippet("ml-loadsql", "Load SQL", "Read a SQL query into a DataFrame.", """
        import pandas as pd

        df = pd.read_sql(${1:"SELECT * FROM table_name"}, ${2:connection})
        """, ["load"], f"{P}/data_loading.py"),
    snippet("ml-loadnumpy", "Load NumPy array", "Load a .npy array saved with numpy.save.", """
        import numpy as np

        values = np.load(${1:"data.npy"})
        """, ["load"], f"{P}/data_loading.py"),
    snippet("ml-loadnumpytext", "Load numeric text", "Load a delimited numeric file with NumPy.", """
        import numpy as np

        values = np.loadtxt(${1:"data.csv"}, delimiter=${2:","})
        """, ["load"], f"{P}/data_loading.py"),
    snippet("ml-loaddataset", "Load dataset by extension", "Choose a CSV, Excel, JSON, or NumPy reader from the suffix.", """
        from pathlib import Path
        import numpy as np
        import pandas as pd

        path = Path(${1:"data.csv"})
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".json":
            df = pd.read_json(path)
        elif path.suffix.lower() == ".npy":
            df = np.load(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        """, ["load"], f"{P}/data_loading.py"),
    snippet("ml-readdataset", "Read a CSV dataset", "Read a CSV dataset and keep a copy of the raw frame.", """
        import pandas as pd

        raw_df = pd.read_csv(${1:"data.csv"})
        df = raw_df.copy()
        """, ["load", "read"], f"{P}/data_loading.py"),
    snippet("ml-loadmultiple", "Load multiple CSV files", "Read every CSV in a folder into a dictionary of frames.", """
        from pathlib import Path
        import pandas as pd

        frames = {path.stem: pd.read_csv(path) for path in Path(${1:"data"}).glob("*.csv")}
        """, ["load"], f"{P}/data_loading.py"),
    snippet("ml-loadclipboard", "Load clipboard", "Read a table that was copied to the clipboard.", """
        import pandas as pd

        df = pd.read_clipboard()
        """, ["load"], f"{P}/data_loading.py"),
    snippet("ml-head", "Head", "Show the first rows.", """
        ${1:df}.head(${2:5})
        """, ["inspect"], f"{P}/data_inspection.py"),
    snippet("ml-tail", "Tail", "Show the last rows.", """
        ${1:df}.tail(${2:5})
        """, ["inspect"], f"{P}/data_inspection.py"),
    snippet("ml-info", "DataFrame info", "Print column dtypes and non-null counts.", """
        ${1:df}.info()
        """, ["inspect"], f"{P}/data_inspection.py"),
    snippet("ml-describe", "Describe", "Summarize numeric and categorical columns.", """
        ${1:df}.describe(include=${2:"all"})
        """, ["inspect"], f"{P}/data_inspection.py"),
    snippet("ml-shape", "Shape", "Read the row and column counts.", """
        print(${1:df}.shape)
        """, ["inspect"], f"{P}/data_inspection.py"),
    snippet("ml-columns", "Columns", "List the column names.", """
        print(${1:df}.columns.tolist())
        """, ["inspect"], f"{P}/data_inspection.py"),
    snippet("ml-dtypes", "Dtypes", "Show the dtype of every column.", """
        print(${1:df}.dtypes)
        """, ["inspect"], f"{P}/data_inspection.py"),
    snippet("ml-valuecounts", "Value counts", "Count categories in one column.", """
        ${1:df}[${2:"column"}].value_counts(dropna=False)
        """, ["inspect"], f"{P}/data_inspection.py"),
    snippet("ml-nunique", "Unique counts", "Count unique values in every column.", """
        ${1:df}.nunique(dropna=False)
        """, ["inspect"], f"{P}/data_inspection.py"),
    snippet("ml-isnull", "Is null", "Boolean missing-value mask using isnull.", """
        ${1:df}.isnull().sum()
        """, ["inspect", "missing"], f"{P}/data_inspection.py"),
    snippet("ml-isna", "Is NA", "Boolean missing-value mask using isna.", """
        ${1:df}.isna().sum()
        """, ["inspect", "missing"], f"{P}/data_inspection.py"),
    snippet("ml-checkmissing", "Check missing values", "Count and percentage of missing values.", """
        missing_counts = ${1:df}.isna().sum()
        missing_percent = (${1:df}.isna().mean() * 100).round(2)
        print(missing_counts)
        print(missing_percent)
        """, ["inspect", "missing"], f"{P}/missing_values.py"),
    snippet("ml-memoryusage", "Memory usage", "Report deep memory usage by column.", """
        ${1:df}.memory_usage(deep=True)
        """, ["inspect"], f"{P}/data_inspection.py"),
    snippet("ml-sample", "Sample rows", "Draw a reproducible sample of rows.", """
        ${1:df}.sample(n=${2:5}, random_state=${3:42})
        """, ["inspect"], f"{P}/data_inspection.py"),
    snippet("ml-missingcount", "Missing counts", "Count missing values in each column.", """
        ${1:df}.isna().sum()
        """, ["missing"], f"{P}/missing_values.py"),
    snippet("ml-missingpercent", "Missing percentage", "Missing values as a percentage of rows.", """
        (${1:df}.isna().mean() * 100).round(2)
        """, ["missing"], f"{P}/missing_values.py"),
    snippet("ml-dropmissing", "Drop missing rows", "Drop rows that contain missing values.", """
        ${1:df} = ${1:df}.dropna()
        """, ["missing"], f"{P}/missing_values.py"),
    snippet("ml-fillna", "Fill NA with a value", "Fill missing values with a constant.", """
        ${1:df} = ${1:df}.fillna(${2:0})
        """, ["missing"], f"{P}/missing_values.py"),
    snippet("ml-imputemean", "Mean imputation", "Fill numeric gaps with the training mean.", """
        means = ${1:X_train}.select_dtypes(include="number").mean()
        ${1:X_train} = ${1:X_train}.fillna(means)
        ${2:X_test} = ${2:X_test}.fillna(means)
        """, ["impute", "missing"], f"{P}/missing_values.py"),
    snippet("ml-imputemedian", "Median imputation", "Fill numeric gaps with the training median.", """
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="median")
        ${1:X_train_imputed} = imputer.fit_transform(${2:X_train})
        ${3:X_test_imputed} = imputer.transform(${4:X_test})
        """, ["impute", "missing"], f"{P}/missing_values.py"),
    snippet("ml-imputemode", "Mode imputation", "Fill gaps with the training mode.", """
        modes = ${1:X_train}.mode(dropna=True).iloc[0]
        ${1:X_train} = ${1:X_train}.fillna(modes)
        ${2:X_test} = ${2:X_test}.fillna(modes)
        """, ["impute", "missing"], f"{P}/missing_values.py"),
    snippet("ml-imputeconstant", "Constant imputation", "Fill gaps with a chosen constant.", """
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="constant", fill_value=${1:0})
        ${2:X_train_imputed} = imputer.fit_transform(${3:X_train})
        ${4:X_test_imputed} = imputer.transform(${5:X_test})
        """, ["impute", "missing"], f"{P}/missing_values.py"),
    snippet("ml-imputeforwardfill", "Forward fill", "Forward-fill each split without using test rows in training.", """
        ${1:X_train} = ${1:X_train}.ffill()
        ${2:X_test} = ${2:X_test}.ffill()
        """, ["impute", "missing"], f"{P}/missing_values.py"),
    snippet("ml-imputebackwardfill", "Backward fill", "Backward-fill each split separately.", """
        ${1:X_train} = ${1:X_train}.bfill()
        ${2:X_test} = ${2:X_test}.bfill()
        """, ["impute", "missing"], f"{P}/missing_values.py"),
    snippet("ml-imputesimple", "SimpleImputer", "Fit SimpleImputer on the training matrix.", """
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy=${1:"mean"})
        ${2:X_train_imputed} = imputer.fit_transform(${3:X_train})
        ${4:X_test_imputed} = imputer.transform(${5:X_test})
        """, ["impute", "missing"], f"{P}/missing_values.py"),
    snippet("ml-imputeknn", "KNN imputation", "Fit KNNImputer on the training matrix.", """
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=${1:5})
        ${2:X_train_imputed} = imputer.fit_transform(${3:X_train})
        ${4:X_test_imputed} = imputer.transform(${5:X_test})
        """, ["impute", "missing"], f"{P}/missing_values.py"),
    snippet("ml-missingindicator", "Missing indicator", "Add missingness flags fit on the training matrix.", """
        from sklearn.impute import MissingIndicator

        indicator = MissingIndicator(features="all")
        ${1:train_flags} = indicator.fit_transform(${2:X_train})
        ${3:test_flags} = indicator.transform(${4:X_test})
        """, ["impute", "missing"], f"{P}/missing_values.py"),
    snippet("ml-imputepipeline", "Impute inside a pipeline", "Median imputation stays inside the model pipeline.", """
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=${1:1000})),
        ])
        pipeline.fit(${2:X_train}, ${3:y_train})
        predictions = pipeline.predict(${4:X_test})
        """, ["impute", "pipeline"], f"{P}/preprocessing_pipeline.py"),
    snippet("ml-duplicated", "Find duplicates", "Mark duplicate rows.", """
        duplicates = ${1:df}[${1:df}.duplicated(keep=${2:"first"})]
        """, ["duplicate"], f"{P}/duplicate_handling.py"),
    snippet("ml-dropduplicates", "Drop duplicates", "Remove duplicate rows and keep the first copy.", """
        ${1:df} = ${1:df}.drop_duplicates()
        """, ["duplicate"], f"{P}/duplicate_handling.py"),
    snippet("ml-dropduplicatessubset", "Drop duplicates by columns", "Remove rows that repeat the selected columns.", """
        ${1:df} = ${1:df}.drop_duplicates(subset=[${2:"column"}])
        """, ["duplicate"], f"{P}/duplicate_handling.py"),
    snippet("ml-dropduplicateskeepfirst", "Keep first duplicate", "Drop duplicates and keep the first record.", """
        ${1:df} = ${1:df}.drop_duplicates(keep="first")
        """, ["duplicate"], f"{P}/duplicate_handling.py"),
    snippet("ml-dropduplicateskeeplast", "Keep last duplicate", "Drop duplicates and keep the last record.", """
        ${1:df} = ${1:df}.drop_duplicates(keep="last")
        """, ["duplicate"], f"{P}/duplicate_handling.py"),
    snippet("ml-stripwhitespace", "Strip whitespace", "Trim text columns.", """
        text_columns = ${1:df}.select_dtypes(include="object").columns
        ${1:df}[text_columns] = ${1:df}[text_columns].apply(lambda column: column.astype("string").str.strip())
        """, ["clean"], f"{P}/inconsistent_data.py"),
    snippet("ml-lowercase", "Lowercase text", "Strip and lowercase text columns.", """
        ${1:df}[${2:"column"}] = ${1:df}[${2:"column"}].astype("string").str.strip().str.lower()
        """, ["clean"], f"{P}/inconsistent_data.py"),
    snippet("ml-uppercase", "Uppercase text", "Strip and uppercase a text column.", """
        ${1:df}[${2:"column"}] = ${1:df}[${2:"column"}].astype("string").str.strip().str.upper()
        """, ["clean"], f"{P}/inconsistent_data.py"),
    snippet("ml-standardizecategory", "Standardize categories", "Replace inconsistent category labels.", """
        mapping = {${1:"pune"}: ${2:"Pune"}, ${3:"delhi"}: ${4:"Delhi"}}
        ${5:df}[${6:"city"}] = ${5:df}[${6:"city"}].replace(mapping)
        """, ["clean"], f"{P}/inconsistent_data.py"),
    snippet("ml-todatetime", "Convert dates", "Parse a column into datetimes.", """
        import pandas as pd

        ${1:df}[${2:"date"}] = pd.to_datetime(${1:df}[${2:"date"}], errors="coerce")
        """, ["clean"], f"{P}/inconsistent_data.py"),
    snippet("ml-astype", "Correct dtypes", "Cast selected columns to new dtypes.", """
        ${1:df} = ${1:df}.astype({${2:"column"}: ${3:"float"}})
        """, ["clean"], f"{P}/inconsistent_data.py"),
    snippet("ml-tonumeric", "Coerce numeric", "Convert a column to numbers and mark invalid entries as missing.", """
        import pandas as pd

        ${1:df}[${2:"column"}] = pd.to_numeric(${1:df}[${2:"column"}], errors="coerce")
        """, ["clean"], f"{P}/inconsistent_data.py"),
    snippet("ml-invalidvalues", "Invalid categories", "Flag values outside an allowed set.", """
        invalid = ~${1:df}[${2:"column"}].isin([${3:"yes"}, ${4:"no"}])
        """, ["clean"], f"{P}/inconsistent_data.py"),
    snippet("ml-rangevalidation", "Range validation", "Flag numeric values outside a closed range.", """
        in_range = ${1:df}[${2:"column"}].between(${3:0}, ${4:100})
        """, ["clean"], f"{P}/inconsistent_data.py"),
    snippet("ml-unitconvert", "Convert units", "Multiply a column by a conversion factor.", """
        ${1:df}[${2:"km"}] = ${1:df}[${3:"miles"}] * ${4:1.60934}
        """, ["clean"], f"{P}/inconsistent_data.py"),
    snippet("ml-renamecolumns", "Rename columns", "Rename columns with a mapping.", """
        ${1:df} = ${1:df}.rename(columns={${2:"old"}: ${3:"new"}})
        """, ["clean"], "implementations/09_Utilities/pandas_utilities.py"),
]
