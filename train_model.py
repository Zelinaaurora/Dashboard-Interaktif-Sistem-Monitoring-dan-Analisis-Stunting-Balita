#Import Library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import kagglehub
import os

# Download dataset
path = kagglehub.dataset_download("rendiputra/stunting-balita-detection-121k-rows")
print("Path to dataset files:", path)

# Cek nama file di folder
print("Files in dataset:", os.listdir(path))

# Load CSV dengan path lengkap
df = pd.read_csv(os.path.join(path, "data_balita.csv"))

df.dropna(inplace=True)

df["Status Gizi"] = (
    df["Status Gizi"]
    .str.lower()
    .str.strip()
)

#Tentukan Feature & Target
X = df.drop("Status Gizi", axis=1)
y = df["Status Gizi"]


#Encoding Data Kategori
encoder = LabelEncoder()

for col in X.select_dtypes(include="object").columns:
    X[col] = encoder.fit_transform(X[col])

y = encoder.fit_transform(y)


#Split Train & Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


#Evaluasi Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance)



#Simpan Model
joblib.dump(model, "rf_stunting_model.pkl")
joblib.dump(X.columns, "feature_columns.pkl")
