from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(app)


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

    # Temporary prediction
    # Later this will be replaced by our ML model
    predicted_rating = 5

    return jsonify({
        "rating": predicted_rating
    })


if __name__ == "__main__":
    app.run(debug=True)