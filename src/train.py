"""Train + tune two classifiers, log everything to MLflow, save the best model."""
import os
import argparse
import joblib
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)

from preprocessing import load_data, clean_data, build_preprocessing_pipeline, split_features_target

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def build_model_grid(scale_pos_weight: float):
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "params": {"clf__C": uniform(0.01, 10)},
            "n_iter": 8,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "params": {
                "clf__n_estimators": randint(100, 400),
                "clf__max_depth": [None, 5, 8, 10, 15],
                "clf__min_samples_leaf": randint(1, 5),
            },
            "n_iter": 12,
        },
        "xgboost": {
            "estimator": XGBClassifier(
                eval_metric="logloss", random_state=42, use_label_encoder=False,
                scale_pos_weight=scale_pos_weight,
            ),
            "params": {
                "clf__n_estimators": randint(100, 400),
                "clf__max_depth": randint(2, 8),
                "clf__learning_rate": uniform(0.01, 0.3),
                "clf__subsample": uniform(0.6, 0.4),
            },
            "n_iter": 12,
        },
    }


def evaluate(model, X_test, y_test, run_dir):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    os.makedirs(run_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    roc_path = os.path.join(run_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    return metrics, cm_path, roc_path


def main(data_path: str, tracking_uri: str = None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-classification")

    df = clean_data(load_data(data_path))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = neg / pos
    print(f"Train set class balance: {neg} negative, {pos} positive "
          f"(scale_pos_weight={scale_pos_weight:.3f})")
    model_grid = build_model_grid(scale_pos_weight)

    best_score = -1
    best_model = None
    best_name = None

    for name, cfg in model_grid.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = Pipeline(steps=[
                ("preprocessor", build_preprocessing_pipeline()),
                ("clf", cfg["estimator"]),
            ])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid = RandomizedSearchCV(
                pipe, cfg["params"], n_iter=cfg["n_iter"], cv=cv,
                scoring="roc_auc", n_jobs=-1, random_state=42,
            )
            grid.fit(X_train, y_train)

            mlflow.log_params(grid.best_params_)
            mlflow.log_param("class_imbalance_handling", "balanced_weight")
            mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
            run_dir = os.path.join(MODEL_DIR, "mlflow_artifacts", name)
            metrics, cm_path, roc_path = evaluate(grid.best_estimator_, X_test, y_test, run_dir)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)
            mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")

            print(f"[{name}] run_id={run.info.run_id} best_params={grid.best_params_} metrics={metrics}")

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = grid.best_estimator_
                best_name = name

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Best model: {best_name} (ROC-AUC={best_score:.4f}) saved to {model_path}")
    return best_model, best_name, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv"))
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    main(args.data, args.tracking_uri)
