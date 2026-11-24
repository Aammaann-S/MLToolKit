"""Short linear-regression practical on synthetic data."""

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=120, n_features=1, noise=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("RMSE", mean_squared_error(y_test, predictions) ** 0.5)
print("R2", r2_score(y_test, predictions))
