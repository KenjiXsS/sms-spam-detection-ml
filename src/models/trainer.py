from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def build_models() -> dict:
    return {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42),
    }


def train_all(models: dict, X_train, y_train) -> dict:
    for model in models.values():
        model.fit(X_train, y_train)
    return models
