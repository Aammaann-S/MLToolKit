"""Short logistic-regression practical on synthetic data."""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=160, n_features=4, n_informative=3, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
