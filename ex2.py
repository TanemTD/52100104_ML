from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

initial_model = RandomForestClassifier(n_estimators=100, random_state=42)
initial_model.fit(X_train, y_train)

y_pred_initial = initial_model.predict(X_test)
accuracy_initial = accuracy_score(y_test, y_pred_initial)
print(f"Initial Test Accuracy: {accuracy_initial}")

X_new = X + np.random.normal(scale=0.1, size=X.shape)
y_new = y

continual_model = initial_model
continual_model.fit(X_new, y_new)

X_production = X + np.random.normal(loc=0.5, scale=0.2, size=X.shape)
y_production = y

y_pred_production = continual_model.predict(X_production)
accuracy_production = accuracy_score(y_production, y_pred_production)
print(f"Production Test Accuracy: {accuracy_production}")
