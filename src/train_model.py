import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import shap
import warnings
import numpy as np

from data_preprocessing import load_data

warnings.filterwarnings("ignore")


# --- Helper class to wrap our pipeline and ensure consistent transformation ---
class WrappedPredictor:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.vectorizer = pipeline.named_steps["tfidf"]
        self.classifier = pipeline.named_steps["clf"]

    def predict_proba(self, raw_texts):
        # Accept only a NumPy array or list of strings.
        # If raw_texts is a NumPy array, flatten it to a list.
        if isinstance(raw_texts, np.ndarray):
            raw_texts = raw_texts.ravel().tolist()
        return self.classifier.predict_proba(self.vectorizer.transform(raw_texts))

    def get_feature_names(self):
        return list(self.vectorizer.get_feature_names_out())


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.
    Returns a dictionary with accuracy, precision, recall, F1, and ROC-AUC.
    """
    y_pred = model.predict(X_test)
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(pd.get_dummies(y_test), y_proba, multi_class="ovr")
        else:
            roc_auc = None
    except Exception:
        roc_auc = None

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "roc_auc": roc_auc,
    }


def train_and_select_model():
    # Load and preprocess data from CSV.
    df = load_data(csv_path="data/mental_health_data.csv")
    X = df["symptoms"]
    y = df["condition"]

    # Split the data into training and testing.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build two pipelines with fixed, limited vocabularies.
    pipeline_lr = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs"))
    ])
    pipeline_rf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, stop_words="english")),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("Training Logistic Regression model...")
    pipeline_lr.fit(X_train, y_train)
    metrics_lr = evaluate_model(pipeline_lr, X_test, y_test)
    print("Logistic Regression metrics:", metrics_lr)

    print("\nTraining Random Forest model...")
    pipeline_rf.fit(X_train, y_train)
    metrics_rf = evaluate_model(pipeline_rf, X_test, y_test)
    print("Random Forest metrics:", metrics_rf)

    # Select the best model based on macro F1 score.
    best_model = pipeline_lr if metrics_lr["f1"] >= metrics_rf["f1"] else pipeline_rf
    best_model_name = "Logistic Regression" if metrics_lr["f1"] >= metrics_rf["f1"] else "Random Forest"
    print("\nSelected Best Model:", best_model_name)

    # --- Wrap the chosen model for consistent transformation ---
    wrapped_model = WrappedPredictor(best_model)

    # Build background data for SHAP using 200 training samples (or fewer).
    background_size = min(200, len(X_train))
    # Get background as a NumPy array with shape (n, 1).
    background_texts = np.array(X_train.sample(n=background_size, random_state=42).tolist()).reshape(-1, 1)
    # Compute numeric representation only for the base value (if needed elsewhere).
    background_matrix = wrapped_model.vectorizer.transform(background_texts.ravel().tolist()).toarray()

    # Build SHAP explainer on the wrapped predict_proba function,
    # passing the background as a 2D NumPy array.
    explainer = shap.KernelExplainer(wrapped_model.predict_proba, background_texts)

    # Use the first test sample to create a SHAP explanation.
    # Convert sample text to a NumPy array.
    sample_text = np.array([X_test.iloc[0]])
    shap_values = explainer.shap_values(sample_text, nsamples=100)

    # For multi-class models, select the explanation for the predicted class.
    # NOTE: For binary classification, shap_values might be a single array.
    predicted_condition = best_model.predict(sample_text)[0]
    classes = list(best_model.named_steps["clf"].classes_)
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            # Multi-class case: choose the array corresponding to the predicted class.
            class_index = classes.index(predicted_condition)
            shap_explanation = np.ravel(shap_values[class_index])
        else:
            # Binary or a single array returned.
            shap_explanation = np.ravel(shap_values[0])
            class_index = 0
    else:
        shap_explanation = np.ravel(shap_values)

    # Retrieve feature names and numeric feature vector.
    feature_names = wrapped_model.get_feature_names()
    sample_features = wrapped_model.vectorizer.transform(sample_text.ravel().tolist()).toarray()[0]
    if len(sample_features) != len(shap_explanation):
        min_len = min(len(sample_features), len(shap_explanation))
        sample_features = sample_features[:min_len]
        feature_names = feature_names[:min_len]

    # Determine the base value using the explainer's expected_value.
    if isinstance(explainer.expected_value, list):
        base_value = explainer.expected_value[class_index]
    else:
        base_value = explainer.expected_value

    # Plot the SHAP force plot using the expected_value as base.
    force_plot = shap.plots.force(
        base_value,             # Expected value as the base value.
        shap_explanation,       # SHAP values.
        sample_features,        # Feature values.
        feature_names=feature_names,
        show=False
    )
    print("SHAP force plot generated (check your browser or the saved HTML).")

    # Save the best model (the full pipeline).
    joblib.dump(best_model, "models/best_model.pkl")
    print("\nBest model saved to models/best_model.pkl")
    return best_model, best_model_name


if __name__ == "__main__":
    train_and_select_model()