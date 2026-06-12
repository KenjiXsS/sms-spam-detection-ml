import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def plot_class_distribution(df: pd.DataFrame, save_path: str = "article/figures/class_distribution.png") -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    df["label"].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "tomato"])
    ax.set_title("Distribuição: Ham vs Spam")
    ax.set_ylabel("Quantidade")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + 0.3, p.get_height() + 20))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrices(models_preds: dict, y_test, save_path: str = "article/figures/confusion_matrix.png") -> None:
    fig, axes = plt.subplots(1, len(models_preds), figsize=(13, 4))
    for ax, (name, pred) in zip(axes, models_preds.items()):
        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curves(models: dict, X_test, y_test, save_path: str = "article/figures/roc.png") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        s = _score(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, s):.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC — Comparação dos Modelos")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_pr_curves(models: dict, X_test, y_test, save_path: str = "article/figures/pr_curve.png") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        s = _score(model, X_test)
        precision, recall, _ = precision_recall_curve(y_test, s)
        ax.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_test, s):.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall — Comparação dos Modelos")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_feature_importance(lr_model, tfidf, save_path: str = "article/figures/feature_importance.png") -> None:
    feature_names = tfidf.get_feature_names_out()
    coefs = lr_model.coef_[0]
    top_spam = pd.Series(coefs, index=feature_names).nlargest(20)
    top_ham = pd.Series(coefs, index=feature_names).nsmallest(20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    top_spam.plot(kind="barh", ax=ax1, color="tomato")
    ax1.set_title("Top 20 palavras → SPAM")
    top_ham.abs().sort_values().plot(kind="barh", ax=ax2, color="steelblue")
    ax2.set_title("Top 20 palavras → HAM")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
