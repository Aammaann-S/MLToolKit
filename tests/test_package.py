import MLToolKit


def test_package_import_exposes_the_shared_catalog():
    assert MLToolKit.__version__ == "1.1.0"
    catalog = MLToolKit.load_catalog()
    triggers = {item["trigger"] for item in catalog["snippets"]}
    assert "ml_loadcsv" in triggers
    assert "ml_knn" in triggers
    assert "ml_kmeans" in triggers
    csv = next(item for item in catalog["snippets"] if item["trigger"] == "ml_loadcsv")
    assert "pd.read_csv" in csv["body"]
    assert "MLToolKit." not in csv["body"]
    assert csv["placeholders"]
