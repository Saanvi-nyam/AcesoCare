from flask import Flask, render_template, request
import joblib
import re
from nltk.corpus import stopwords

# Load saved model and vectorizer
model = joblib.load("knn_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Flask app
app = Flask(__name__)

# Load stopwords
stop_words = set(stopwords.words('english'))

# Preprocess function (regex-based, no punkt needed)
def preprocess_text(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())  # keep only a-z words
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_disease = None
    if request.method == "POST":
        symptom = request.form["symptom"]
        preprocessed = preprocess_text(symptom)
        symptom_tfidf = vectorizer.transform([preprocessed])
        predicted_disease = model.predict(symptom_tfidf)[0]

    return render_template("index.html", prediction=predicted_disease)

if __name__ == "__main__":
    app.run(debug=True)
