from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
import joblib

vectorizer = joblib.load("ml/tfidf_vectorizer.pkl")
model = joblib.load("ml/tuned_logistic_model.pkl")

@app.route("/")
def home():
    return jsonify({
        "message": "Amazon Rating Predictor Backend is running!"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    review = data.get("review_text", "")

    if not review.strip():
        return jsonify({
            "error": "Please enter a review."
        }), 400

    # ML model prediction
    X = vectorizer.transform([review])
    prediction = model.predict(X)[0]

    return jsonify({
        "prediction": int(prediction)
})

if __name__ == "__main__":
    app.run(debug=True)

