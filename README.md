# 📊 RAID & Decision NLP Classifier

An NLP-based classification system that identifies key project management statements such as Risks, Assumptions, Dependencies, and Decisions from unstructured text.

---

## 🚀 Project Overview

In real-world projects, important information is often buried in:

meeting notes
status updates
emails
documentation

This project uses Natural Language Processing (NLP) to automatically classify such text into meaningful categories to improve project tracking and decision-making.

---

## 🎯 Objective

To build a machine learning model that can classify a given sentence into:

Risk → potential problems or uncertainties
Assumption → beliefs taken as true without confirmation
Dependency → reliance on external factors or teams
Decision → confirmed choices or actions taken
Other → general project statements

---

## 🧠 Why RAID?

RAID is a common project management framework:

R → Risks
A → Assumptions
I → Issues
D → Dependencies

This project extends RAID by also including:
👉 Decision detection, which is highly valuable in real project workflows.

---

## 🛠️ Tech Stack
Python
pandas
scikit-learn
TF-IDF Vectorization
Logistic Regression
Streamlit (for UI)
Jupyter Notebook (for experimentation)

---

## 📂 Project Structure
raid-nlp-classifier/
│
├── data/
│   └── raw/                 # Dataset
│
├── notebooks/              # Experimentation
│
├── src/
│   ├── train_baseline.py   # Model training
│   ├── predict.py          # Inference
│
├── app/
│   └── streamlit_app.py    # Web app
│
├── results/                # Saved model
│
├── requirements.txt
├── README.md
└── .gitignore

---

## ⚙️ How It Works
Text input is passed into the model
TF-IDF converts text into numerical features
Logistic Regression predicts the class
Output includes:
predicted label
confidence scores

---

## 📈 Example

Input:

"The deployment depends on approval from the security team."

Output:

Prediction: Dependency
Confidence:
Dependency: 0.55
Risk: 0.11
Assumption: 0.11
Decision: 0.11
Other: 0.11

---

## ▶️ How to Run
1. Clone the repository
git clone https://github.com/your-username/raid-nlp-classifier.git
cd raid-nlp-classifier
2. Create virtual environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Train the model
python src/train_baseline.py
5. Run the app
python -m streamlit run app/streamlit_app.py

---

## 📊 Current Model
Model: TF-IDF + Logistic Regression
Type: Multi-class classification
Dataset: Custom labeled project statements

---

## 🔍 Limitations
Small dataset (manual labeling)
Model relies on simple text patterns
Limited understanding of complex context

---

## 🚀 Future Improvements
Fine-tune DistilBERT / BERT
Add Issue classification (complete RAID)
Improve dataset with real-world data
Support multi-sentence inputs
Deploy app online (Streamlit Cloud / Render)
Add explainability (highlight important words)

---

## 💡 Key Learnings
Built an end-to-end NLP pipeline
Implemented baseline ML model for text classification
Learned feature extraction using TF-IDF
Developed a simple interactive UI using Streamlit
Structured a real-world ML project for GitHub

---

## 🤝 Contribution

This is a personal learning project, but suggestions and improvements are welcome!

---

## 📌 Author

Washifur Rahman
