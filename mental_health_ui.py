import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

# WrappedPredictor ensures proper text transformation using the saved pipeline.
class WrappedPredictor:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.vectorizer = pipeline.named_steps["tfidf"]
        self.classifier = pipeline.named_steps["clf"]

    def predict_proba(self, raw_texts):
        if isinstance(raw_texts, np.ndarray):
            raw_texts = raw_texts.ravel().tolist()
        return self.classifier.predict_proba(self.vectorizer.transform(raw_texts))

    def get_feature_names(self):
        return list(self.vectorizer.get_feature_names_out())

# Cache the model loading using st.cache_resource (replacing deprecated st.cache)
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    return model, WrappedPredictor(model)

model, wrapped_model = load_model()

# Streamlit UI Components
st.title("Self‐Analysis Mental Health Predictor")
st.write("Enter your symptoms below to predict possible mental health conditions.")

input_text = st.text_area("Symptoms:", "e.g., feeling down, lack of energy, trouble sleeping")

if st.button("Predict"):
    if input_text.strip() == "":
        st.error("Please enter valid symptoms.")
    else:
        # Generate prediction and display condition and probabilities
        predicted_condition = model.predict([input_text])[0]
        st.subheader("Predicted Mental Health Condition:")
        st.write(predicted_condition)

        proba = model.predict_proba([input_text])[0]
        classes = list(model.named_steps["clf"].classes_)
        st.subheader("Prediction Probabilities:")
        for condition, p in zip(classes, proba):
            st.write(f"**{condition}**: {p:.2f}")

        try:
            # Initialize SHAP JavaScript rendering
            shap.initjs()
            
            # Reshape the input for consistency with the training script
            background = np.array([input_text], dtype=object).reshape(-1, 1)
            explainer = shap.KernelExplainer(wrapped_model.predict_proba, background)

            sample_text = np.array([input_text], dtype=object).reshape(-1, 1)
            shap_values = explainer.shap_values(sample_text, nsamples=100)

            if isinstance(shap_values, list) and len(shap_values) > 1:
                try:
                    class_index = classes.index(predicted_condition)
                except ValueError:
                    class_index = 0
                shap_explanation = np.ravel(shap_values[class_index])
            else:
                class_index = 0
                shap_explanation = np.ravel(shap_values)

            expected_vals = np.atleast_1d(explainer.expected_value)
            if len(expected_vals) > 1 and class_index < len(expected_vals):
                base_value = expected_vals[class_index]
            else:
                base_value = expected_vals[0]

            input_matrix = wrapped_model.vectorizer.transform([input_text]).toarray()
            sample_features = input_matrix[0]
            feature_names = wrapped_model.get_feature_names()
            if len(sample_features) != len(shap_explanation):
                min_len = min(len(sample_features), len(shap_explanation))
                sample_features = sample_features[:min_len]
                feature_names = feature_names[:min_len]

            sample_features_df = pd.DataFrame([sample_features], columns=feature_names)

            # IMPORTANT: Set matplotlib to False so that an interactive JavaScript force plot is generated.
            force_plot = shap.force_plot(
                base_value,
                shap_explanation,
                sample_features_df,
                feature_names=feature_names,
                show=False,
                matplotlib=False
            )
            # Save the force plot as an HTML file and embed it in the UI.
            temp_html = "temp_shap.html"
            shap.save_html(temp_html, force_plot)
            with open(temp_html, "r", encoding="utf-8") as f:
                shap_html = f.read()

            st.subheader("SHAP Explanation:")
            components.html(shap_html, height=600, scrolling=True)
        except Exception as e:
            st.error(f"Could not generate SHAP explanation. Error: {e}")