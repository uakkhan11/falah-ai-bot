import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report

CSV_PATH = "your_training_data.csv"  # Update path as needed
MODEL_SAVE_PATH = "model.pkl"
TARGET_COLUMN = "outcome"
FEATURES = ["rsi", "atr", "adx", "ema10", "ema21", "volumechange"]

def train_and_save_model():
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Model file '{MODEL_SAVE_PATH}' already exists. Loading model...")
        model = joblib.load(MODEL_SAVE_PATH)
        return model

    print("Training new model from scratch...")

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.lower() for c in df.columns]
    df.dropna(subset=FEATURES + [TARGET_COLUMN], inplace=True)

    X = df[FEATURES]
    y = df[TARGET_COLUMN].astype(int)

    # Sample for faster hyperparameter tuning
    X_sample, _, y_sample, _ = train_test_split(
        X, y, test_size=0.9, stratify=y, random_state=42)

    param_dist = {
        "n_estimators": [100, 150],
        "max_depth": [None, 10],
        "min_samples_split": [2],
        "class_weight": ["balanced"],
    }

    from sklearn.model_selection import RandomizedSearchCV
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    random_search.fit(X_sample, y_sample)
    print(f"Best params: {random_search.best_params_}")

    final_model = RandomForestClassifier(**random_search.best_params_, random_state=42)
    final_model.fit(X, y)

    cv_scores = cross_val_score(final_model, X, y, cv=5, scoring="f1")
    print(f"5-fold CV F1 score: {cv_scores.mean():.4f}")

    y_pred = final_model.predict(X)
    print("Classification report on training data:")
    print(classification_report(y, y_pred))

    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    return final_model

if __name__ == "__main__":
    train_and_save_model()
