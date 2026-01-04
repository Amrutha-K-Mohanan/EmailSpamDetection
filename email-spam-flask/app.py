from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()

def preprocess(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""

    if request.method == "POST":
        email = request.form["email"]
        processed_email = preprocess(email)
        vector = cv.transform([processed_email])
        result = model.predict(vector)

        prediction = "Spam 🚫" if result[0] == 1 else "Not Spam ✅"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
