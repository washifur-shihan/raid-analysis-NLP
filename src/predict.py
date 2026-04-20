import joblib

MODEL_PATH = "results/baseline_model.joblib"

def predict_text(text):
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    scores = dict(zip(classes, probabilities))
    return prediction, scores

if __name__ == "__main__":
    sample = "The launch depends on final approval from finance."
    label, scores = predict_text(sample)
    print("Prediction:", label)
    print("Scores:", scores)