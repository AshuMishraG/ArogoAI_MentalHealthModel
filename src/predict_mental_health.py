import sys
import joblib
import shap
import numpy as np
import pandas as pd

# Minimal WrappedPredictor to enforce the correct input type.
class WrappedPredictor:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.vectorizer = pipeline.named_steps["tfidf"]
        self.classifier = pipeline.named_steps["clf"]

    def predict_proba(self, raw_texts):
        # If the input is a NumPy array, flatten it to a list.
        if isinstance(raw_texts, np.ndarray):
            raw_texts = raw_texts.ravel().tolist()
        # Now, raw_texts is a list of strings.
        return self.classifier.predict_proba(self.vectorizer.transform(raw_texts))

    def get_feature_names(self):
        return list(self.vectorizer.get_feature_names_out())

def main():
    # Load the best saved model.
    model = joblib.load("models/best_model.pkl")
    wrapped_model = WrappedPredictor(model)

    # Accept user input via command line or prompt.
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = input("Enter your symptoms (as text): ")

    # Generate a prediction.
    predicted_condition = model.predict([input_text])[0]
    print("\nPredicted Mental Health Condition:", predicted_condition)

    # Display prediction probabilities.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([input_text])[0]
        classes = list(model.named_steps["clf"].classes_)
        print("\nPrediction Probabilities:")
        for condition, p in zip(classes, proba):
            print(f"  {condition}: {p:.2f}")

    # --- Generate the SHAP Explanation ---
    try:
        # In the training script we converted background texts to a 2D array.
        # Here we do the same for consistency.
        background = np.array([input_text], dtype=object).reshape(-1, 1)
        # Create the KernelExplainer using the wrapped predict_proba.
        explainer = shap.KernelExplainer(wrapped_model.predict_proba, background)

        # Create the sample input in the same 2D shape.
        sample_text = np.array([input_text], dtype=object).reshape(-1, 1)
        shap_values = explainer.shap_values(sample_text, nsamples=100)
        print("\nGenerating SHAP explanation for the prediction...")
        shap.initjs()

        # Determine the predicted class index.
        classes = list(model.named_steps["clf"].classes_)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            try:
                class_index = classes.index(predicted_condition)
            except ValueError:
                class_index = 0
            shap_explanation = np.ravel(shap_values[class_index])
        else:
            class_index = 0
            shap_explanation = np.ravel(shap_values)

        # Robustly determine the base (expected) value.
        expected_vals = np.atleast_1d(explainer.expected_value)
        if len(expected_vals) > 1 and class_index < len(expected_vals):
            base_value = expected_vals[class_index]
        else:
            base_value = expected_vals[0]

        # Obtain the numeric feature vector from the TF-IDF transformer.
        # (This is identical to what we do in train_model.py.)
        input_matrix = wrapped_model.vectorizer.transform([input_text]).toarray()
        sample_features = input_matrix[0]  # 1D array of feature values.
        feature_names = wrapped_model.get_feature_names()
        if len(sample_features) != len(shap_explanation):
            min_len = min(len(sample_features), len(shap_explanation))
            sample_features = sample_features[:min_len]
            feature_names = feature_names[:min_len]

        # IMPORTANT: Convert the numeric feature row into a DataFrame.
        # SHAP's force_plot expects a 2D structure with proper column names.
        sample_features_df = pd.DataFrame([sample_features], columns=feature_names)

        # Generate the force plot using the explainer's expected value as the base.
        force_plot = shap.force_plot(
            base_value,             # Base (expected) value.
            shap_explanation,       # SHAP values.
            sample_features_df,     # 2D DataFrame of feature values.
            feature_names=feature_names,
            show=False
        )
        shap.save_html("shap_explanation.html", force_plot)
        print("SHAP explanation saved as 'shap_explanation.html'. Open it in a browser to view.")
    except Exception as e:
        print("Could not generate SHAP explanation. Error:", e)

if __name__ == "__main__":
    main()