import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Import dataset
df = pd.read_csv("cancer_risk_dataset.csv")

# Memisahkan "feature" dan "target"
X = df.drop(columns="cancer_risk")
y = df["cancer_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Membuat model dengan menerapkan algoritma Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Menampilkan akurasi model
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {score*100:.2f}%.")