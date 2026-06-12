import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar


def evaluate(model, X_test, y_test, name: str = "") -> None:
    pred = model.predict(X_test)
    print(f"=== {name} ===")
    print(classification_report(y_test, pred, target_names=["ham", "spam"]))
    return pred


def comparison_table(models_preds: dict, y_test) -> pd.DataFrame:
    rows = []
    for name, pred in models_preds.items():
        rows.append(
            {
                "Modelo": name,
                "Acurácia": f"{accuracy_score(y_test, pred):.4f}",
                "F1 (spam)": f"{f1_score(y_test, pred):.4f}",
                "F1 macro": f"{f1_score(y_test, pred, average='macro'):.4f}",
            }
        )
    return pd.DataFrame(rows).set_index("Modelo")


def cross_validate(model, X_train, y_train, n_splits: int = 5) -> tuple:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    return scores.mean(), scores.std(), scores


def mcnemar_test(pred_a, pred_b, y_true) -> tuple:
    b = int(((pred_a == y_true) & (pred_b != y_true)).sum())
    c = int(((pred_a != y_true) & (pred_b == y_true)).sum())
    result = _mcnemar([[0, b], [c, 0]], exact=False, correction=True)
    return result.pvalue, b, c


def auc_scores(models: dict, X_test, y_test) -> dict:
    out = {}
    for name, model in models.items():
        score = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )
        out[name] = {
            "auc_roc": roc_auc_score(y_test, score),
            "avg_precision": average_precision_score(y_test, score),
        }
    return out
