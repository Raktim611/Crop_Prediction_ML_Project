import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Crop_recommendation.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.txt")

FEATURES = [
    "N", "P", "K",
    "temperature", "humidity", "ph", "rainfall"
]
TARGET = "label"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Basic validation
    if data[required].isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values.")
    if data.duplicated().sum() > 0:
        print(f"Warning: {data.duplicated().sum()} duplicate rows found.")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    }

    results = []
    fitted_models = {}

    print("\nMODEL COMPARISON")
    print("-" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        results.append((name, acc))
        fitted_models[name] = model

        print(f"{name:22s}: {acc:.4f}")

    # Select highest test accuracy
    best_name, best_accuracy = max(results, key=lambda x: x[1])
    best_model = fitted_models[best_name]

    # Save model + metadata
    bundle = {
        "model": best_model,
        "features": FEATURES,
        "target": TARGET,
        "model_name": best_name,
        "test_accuracy": best_accuracy,
        "classes": list(best_model.classes_)
    }

    joblib.dump(bundle, MODEL_PATH)

    best_pred = best_model.predict(X_test)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("CROP RECOMMENDATION MODEL RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset rows: {len(data)}\n")
        f.write(f"Number of classes: {y.nunique()}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Testing samples: {len(X_test)}\n\n")
        f.write("MODEL ACCURACY\n")
        f.write("-" * 60 + "\n")
        for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
            f.write(f"{name}: {acc:.6f}\n")

        f.write("\n")
        f.write(f"BEST MODEL: {best_name}\n")
        f.write(f"TEST ACCURACY: {best_accuracy:.6f}\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 60 + "\n")
        f.write(classification_report(y_test, best_pred))

        if hasattr(best_model, "feature_importances_"):
            f.write("\nFEATURE IMPORTANCE\n")
            f.write("-" * 60 + "\n")
            for feature, importance in sorted(
                zip(FEATURES, best_model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            ):
                f.write(f"{feature}: {importance:.6f}\n")

    print("\n" + "=" * 60)
    print(f"BEST MODEL : {best_name}")
    print(f"TEST ACCURACY: {best_accuracy:.4%}")
    print(f"Saved to    : {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
