import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("ml/tuned_logistic_model.pkl")
vectorizer = joblib.load("ml/tfidf_vectorizer.pkl")


def predict_rating(review_text):
    # Convert review text into TF-IDF features
    text_vector = vectorizer.transform([review_text])

    # Predict
    prediction = model.predict(text_vector)[0]

    return int(prediction)


# Test prediction
review = "This product is excellent and I really love it!"

result = predict_rating(review)

print("Review:", review)
print("Prediction:", result)