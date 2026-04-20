import streamlit as st
import joblib

MODEL_PATH = "results/baseline_model.joblib"

st.title("RAID Statement Classifier")
st.write("Classify project statements into Risk, Assumption, Dependency, Decision, or Other.")

model = joblib.load(MODEL_PATH)

user_input = st.text_area("Enter a project statement:")

if st.button("Classify"):
    if user_input.strip():
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]
        classes = model.classes_

        st.subheader("Prediction")
        st.write(prediction)

        st.subheader("Confidence Scores")
        for label, score in zip(classes, probabilities):
            st.write(f"{label}: {score:.4f}")
    else:
        st.warning("Please enter some text.")